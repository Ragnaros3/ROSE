"""Mask-aware hierarchical spectral mapper and training losses."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ResidualMLPBlock(nn.Module):
    """Residual multilayer block used by each view encoder."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(width)),
            nn.Linear(int(width), int(width)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(width), int(width)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return (values + self.net(values)) / math.sqrt(2.0)


class HierarchicalViewEncoderDecoder(nn.Module):
    """Hierarchical encoder and observed-view decoder for one view."""

    def __init__(
        self,
        input_dim: int,
        width: int = 128,
        latent_dim: int = 32,
        blocks: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if input_dim < 1 or width < 1 or latent_dim < 1 or blocks < 0:
            raise ValueError("Network dimensions must be positive and blocks nonnegative.")
        projection_width = max(64, int(latent_dim))
        self.stem = nn.Sequential(
            nn.Linear(int(input_dim), int(width)),
            nn.LayerNorm(int(width)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualMLPBlock(int(width), float(dropout))
                for _ in range(int(blocks))
            ]
        )
        self.low_projection = nn.Sequential(
            nn.Linear(int(width), projection_width),
            nn.LayerNorm(projection_width),
            nn.ReLU(inplace=True),
            nn.Linear(projection_width, int(latent_dim)),
        )
        self.high_refinement = nn.Sequential(
            nn.LayerNorm(int(latent_dim)),
            nn.Linear(int(latent_dim), 2 * int(latent_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * int(latent_dim), int(latent_dim)),
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(latent_dim), int(width)),
            nn.ReLU(inplace=True),
            nn.Linear(int(width), int(input_dim)),
        )

    def encode(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        body = self.blocks(self.stem(values))
        low = F.normalize(self.low_projection(body), dim=1)
        high = F.normalize(low + self.high_refinement(low), dim=1)
        return low, high

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


class S2HierarchicalSpectralMapper(nn.Module):
    """Frozen S2 architecture used for one-way spectral transfer."""

    def __init__(
        self,
        view_dims: list[int],
        n_clusters: int,
        width: int = 128,
        latent_dim: int = 32,
        blocks: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if not view_dims or any(int(value) < 1 for value in view_dims):
            raise ValueError("view_dims must contain positive dimensions.")
        if n_clusters < 1:
            raise ValueError("n_clusters must be positive.")

        self.n_views = len(view_dims)
        self.views = nn.ModuleList(
            [
                HierarchicalViewEncoderDecoder(
                    input_dim=int(view_dim),
                    width=int(width),
                    latent_dim=int(latent_dim),
                    blocks=int(blocks),
                    dropout=float(dropout),
                )
                for view_dim in view_dims
            ]
        )
        fusion_input_dim = int(latent_dim) * self.n_views + self.n_views
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, int(latent_dim)),
            nn.LayerNorm(int(latent_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(latent_dim), int(latent_dim)),
        )
        self.spectral_head = nn.Linear(
            int(latent_dim), int(n_clusters), bias=False
        )

    def forward(
        self,
        views: list[torch.Tensor],
        mask: torch.Tensor,
    ) -> dict[str, Any]:
        if len(views) != self.n_views:
            raise ValueError("The number of input views does not match the mapper.")
        if mask.ndim != 2 or mask.shape[1] != self.n_views:
            raise ValueError("mask has an incompatible shape.")
        if any(view.ndim != 2 or view.shape[0] != mask.shape[0] for view in views):
            raise ValueError("Each view must contain one row per mask row.")

        masked_views = [
            view * mask[:, view_id : view_id + 1]
            for view_id, view in enumerate(views)
        ]
        encoded = [
            module.encode(view)
            for module, view in zip(self.views, masked_views)
        ]
        low_latents = [pair[0] for pair in encoded]
        high_latents = [pair[1] for pair in encoded]
        included_latents = [
            latent * mask[:, view_id : view_id + 1]
            for view_id, latent in enumerate(high_latents)
        ]
        fusion_input = torch.cat(included_latents + [mask], dim=1)
        fused = F.normalize(self.fusion(fusion_input), dim=1)
        scores = self.spectral_head(fused)
        reconstructions = [
            module.decode(latent)
            for module, latent in zip(self.views, high_latents)
        ]
        return {
            "observed_latents": high_latents,
            "low_latents": low_latents,
            "fused": fused,
            "spectral": scores,
            "reconstructions": reconstructions,
        }


def normalize_spectral_tensor(values: torch.Tensor) -> torch.Tensor:
    """Column-center scores and normalize each sample row."""

    centered = values - values.mean(dim=0, keepdim=True)
    return F.normalize(centered, dim=1)


def procrustes_aligned_loss(
    student_scores: torch.Tensor,
    spectral_target: torch.Tensor,
) -> torch.Tensor:
    """Align normalized student scores to the final frozen target.

    ``spectral_target`` is the final twice-normalized target constructed by
    :func:`rose.spectral.build_spectral_target`.  It is therefore used as
    supplied; only the raw student scores are centered and row-normalized here.
    """

    if student_scores.shape != spectral_target.shape:
        raise ValueError("Student scores and target must have the same shape.")
    student = normalize_spectral_tensor(student_scores)
    target = spectral_target
    with torch.no_grad():
        left, _singular_values, right_t = torch.linalg.svd(
            student.T @ target,
            full_matrices=False,
        )
        rotation = left @ right_t
    return F.mse_loss(student @ rotation, target)


def observed_reconstruction_loss(
    outputs: dict[str, Any],
    views: list[torch.Tensor],
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute reconstruction error only where a view is observed."""

    reconstructions = outputs.get("reconstructions")
    if not isinstance(reconstructions, list) or len(reconstructions) != len(views):
        raise TypeError("Expected one reconstruction tensor per view.")
    weighted_losses: list[torch.Tensor] = []
    for view_id, (reconstruction, target) in enumerate(
        zip(reconstructions, views)
    ):
        per_sample = F.mse_loss(
            reconstruction, target, reduction="none"
        ).mean(dim=1)
        weighted_losses.append(per_sample * mask[:, view_id])
    stacked = torch.stack(weighted_losses, dim=1)
    return stacked.sum() / mask.sum().clamp_min(1.0)


def observed_cross_view_loss(
    outputs: dict[str, Any],
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute latent agreement over co-observed sample-view pairs."""

    latents = outputs.get("observed_latents")
    if not isinstance(latents, list):
        raise TypeError("Expected a list of observed-view latent tensors.")
    losses: list[torch.Tensor] = []
    for left_id in range(len(latents)):
        for right_id in range(left_id + 1, len(latents)):
            valid = (mask[:, left_id] > 0.5) & (mask[:, right_id] > 0.5)
            if bool(valid.any()):
                cosine = F.cosine_similarity(
                    latents[left_id][valid],
                    latents[right_id][valid],
                    dim=1,
                )
                losses.append((1.0 - cosine).mean())
    if not losses:
        return torch.zeros((), dtype=mask.dtype, device=mask.device)
    return torch.stack(losses).mean()


def joint_spectral_loss(
    mapper: nn.Module,
    views: list[torch.Tensor],
    mask: torch.Tensor,
    spectral_target: torch.Tensor,
    lambda_spec: float = 1.0,
    lambda_rec: float = 0.10,
    lambda_cv: float = 0.05,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Evaluate the frozen weighted training objective."""

    outputs = mapper(views, mask)
    scores = outputs.get("spectral")
    if not isinstance(scores, torch.Tensor):
        raise TypeError("Mapper output must contain a spectral tensor.")
    loss_spec = procrustes_aligned_loss(scores, spectral_target)
    loss_rec = observed_reconstruction_loss(outputs, views, mask)
    loss_cv = observed_cross_view_loss(outputs, mask)
    loss_total = (
        float(lambda_spec) * loss_spec
        + float(lambda_rec) * loss_rec
        + float(lambda_cv) * loss_cv
    )
    return outputs, {
        "loss_total": loss_total,
        "loss_spec": loss_spec,
        "loss_rec": loss_rec,
        "loss_cv": loss_cv,
    }
