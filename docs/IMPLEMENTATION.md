# ROSE implementation

The implementation follows main-text Sections 3–5 and Supplementary Section S1.3. Numerical settings and seed rules are collected in `config/frozen_method.json`.

## Data, graph and spectral target

Observed samples are standardized view by view using feature means and population variances. Constant features use scale one. Missing rows are zero-filled and gated by the observation mask. `rose/protocol.py` separates feature loading from label loading.

`rose/graph.py` constructs co-observed neighborhoods and reweights their existing support. The candidate pool is 45, adaptive neighbor counts lie between 10 and 35 with reference count 25, local scale uses neighbor 7, and the similarity floor is 0.01. Observed-only reconstruction uses up to five neighbors. The positive context and view-quality factors are bounded by 0.5 and 1.5; the outer rank factor is 0.5. The view-quality activation ratio is 3.

Context strata use source coverage, unordered endpoint-degree terciles at quantiles 0.33 and 0.66, and mask-pattern equality. The minimum stratum size is 50 for fewer than 5,000 support edges and 100 otherwise. Backoff follows coverage–degree–pattern, coverage–degree, coverage, then global.

`rose/spectral.py` extracts the largest-algebraic normalized-adjacency eigenmodes with tolerance 1e-4 and a normalized Gaussian starting vector determined by the protocol seed. Modes are sorted by decreasing eigenvalue; each sign is fixed by its largest-magnitude entry. Dense fallback is restricted to at most 5,000 samples.

For the ordered K+1-dimensional basis, float32 row scaling precedes selection of the first K coordinates, giving T. The row-scaling threshold is 10 times float32 machine epsilon (approximately 1.1921e-6). Let N center columns and divide each row by its norm floored at 1e-12. The first normalization is U0=N(T); the mapper loss target is U=N(U0). The graph and target remain fixed during mapper and corrector fitting.

Other numerical values are epsilon=1e-8, edge-kernel epsilon=1e-12 and local-scale floor=1e-6. Reconstruction weights become uniform when their numerical sum is at most epsilon. A view without an eligible co-observed partition comparison uses quality score one.

## Mapper and checkpoint selection

Each view uses a Linear–LayerNorm–ReLU–Dropout stem, two pre-normalized residual MLP blocks, a low projection and a residual high refinement. Each residual block divides its input-plus-update by sqrt(2). Fusion concatenates masked view representations and the observation mask. The spectral head is bias-free. Dimensions, dropout, loss weights and full-batch AdamW settings are in the configuration.

The mapper records epoch 0 and every fifth epoch through 120. Valid candidates must have epoch at least 20, observed-support validity, finite diagnostics, exactly K active clusters, normalized cluster entropy at least 0.60, largest-cluster fraction at most 0.60, effective-rank ratio at least 0.65, and mean coordinate variance at least 1e-5.

For column-centered Y, let a be its positive squared singular values normalized by their sum and e(Y)=exp(-sum(a*log(a))). The rank ratio is

```text
e(N(S)) / max(e(U0), 1e-12)
```

It uses U0, whereas the spectral loss uses U. Agreement is arithmetic-normalized NMI between the checkpoint partition and fixed graph partition, without class labels.

The selector in `rose/selector.py` applies these rules:

1. If there is no valid candidate, select the minimum-spectral-loss state from all recorded checkpoints, including epoch 0. Loss ties select the earlier epoch.
2. If the co-observation giant-component ratio is at least 0.98, choose the minimum-loss valid candidate.
3. Otherwise retain candidates with agreement at least 0.95 times the maximum. Prefer candidates whose entropy is within 0.010 of the maximum and largest-cluster fraction within 0.030 of the minimum, then choose the earliest. If this balanced subset is empty, use the earliest in the agreement band; if the band is empty, use the minimum-loss valid candidate.
4. For a provisional epoch at least 100, search earlier valid candidates with loss at most `best_loss + 0.25*max(first_valid_loss-best_loss,0)` and agreement at least 0.88 times the maximum. Replace the provisional state by the earliest qualifying candidate only if its largest-cluster fraction has risen by at least 0.05 and its entropy has fallen by at least 0.008 relative to that candidate.

The agreement, loss-band and drift comparisons use tolerance 1e-12. The selected mapper is frozen before the correction decision.

## Bounded correction and evaluation

The corrector uses counterfactual view drops on complete samples. The supplied missingness settings retain complete samples for this decision. Admission uses 1,000 bootstrap replicates and the lower 2.5th percentile of the response statistic, with threshold 2*rho=0.20. The corrector state is taken after 40 updates.

`rose/correction.py` projects the raw update orthogonally to the base scores and bounds its norm by rho=0.10 times the base-score norm, with the explicit zero branch. Complete samples receive zero direct residual. The final normalized scores are clustered with 20 K-means restarts. The protocol seed controls final clustering; mapper, corrector and bootstrap seed formulas are in the configuration. True labels enter only optional final metric evaluation.
