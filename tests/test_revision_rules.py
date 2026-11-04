"""Small rule checks: no fitting, graph construction or benchmark data."""
from __future__ import annotations

import ast
import inspect
import unittest
from unittest import mock

import numpy as np
import torch

from rose.spectral import SpectralTarget, normalize_coordinates, finalize_spectral_target
from rose.training import _checkpoint_diagnostics, _effective_rank, train_mapper


class RevisionRuleTests(unittest.TestCase):
    def test_rank_denominator_uses_first_normalization(self) -> None:
        coordinates = np.array([[4., 1., 0.], [2., 1., 1.], [0., 3., 0.],
                                [1., 0., 1.], [-2., 1., 1.], [0., 0., -4.]],
                               dtype=np.float32)
        first = normalize_coordinates(coordinates)
        target = finalize_spectral_target(first)
        # This fixture distinguishes the two denominators; it is not an
        # empirical experiment or a new model prediction.
        self.assertGreater(abs(_effective_rank(first)-_effective_rank(target)), 1e-4)
        partition = np.array([0, 0, 1, 1, 2, 2])
        bundle = SpectralTarget(coordinates, coordinates, first, target,
                                np.array([1., .8, .7]), partition)
        losses = {k:torch.tensor(.2) for k in ['loss_total','loss_spec','loss_rec','loss_cv']}
        with mock.patch('rose.training.kmeans_partition', return_value=partition):
            diag, _ = _checkpoint_diagnostics(20, losses, coordinates, bundle, 3, 1, True)
        expected = _effective_rank(normalize_coordinates(coordinates))/_effective_rank(first)
        self.assertAlmostEqual(diag.effective_rank_ratio, expected, places=6)
        wrong = _effective_rank(first)/_effective_rank(target)
        self.assertGreater(abs(diag.effective_rank_ratio-wrong), 1e-4)

    def test_initial_diagnostics_precede_optimizer_updates(self) -> None:
        # Inspect the control flow rather than invoking a training pipeline.
        tree = ast.parse(inspect.getsource(train_mapper))
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Name) and n.target.id=='epoch')
        self.assertIsInstance(loop.iter, ast.Call)
        self.assertEqual(ast.literal_eval(loop.iter.args[0]), 0)
        guard = loop.body[0]
        self.assertIsInstance(guard, ast.If)
        self.assertEqual(ast.unparse(guard.test), 'epoch > 0')
        self.assertIn('optimizer.step()', ast.unparse(guard))
        self.assertTrue(any('endpoints[int(epoch)]' in ast.unparse(n)
                            for n in loop.body[1:]))


if __name__ == '__main__':
    unittest.main()
