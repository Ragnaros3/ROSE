# Experimental protocol

These settings correspond to the revised manuscript's experimental setup and Supplementary Sections S1 and S4. This file documents the evaluated procedures; `run_rose.py` runs the standard 120-epoch ROSE configuration.

## Shared benchmark protocol

The five processed benchmarks are BDGP, Fashion, Handwritten, Hdigit and MNIST-USPS. Their exact matrix identities and view dimensions are in `data_sources.csv`. Each setting uses incomplete-sample fractions 0.1, 0.3, 0.5, 0.7 and 0.9, with seeds 1–3 and a supplied cluster count K. Each incomplete sample loses one view.

Full-view (FV) methods retain all original views. Reduced-view (RV) adaptations regenerate masks on their retained views using the same protocol seed. ROSE standardizes observed samples only; baseline preprocessing is specified below. All reported scores were computed in the authors' experiments. Some upstream interfaces compute metrics during fitting; the stated output-selection rules determine which output enters the comparison.

## Eight comparison methods (Supplementary Table S1)

| Method and source | Inputs and preprocessing | Training and output |
|---|---|---|
| [COMPLETER](https://github.com/XLearning-SCU/2021-CVPR-Completer) | First two Fashion/Handwritten views; native two-view inputs elsewhere. Loader values retained as float32; reshape/transpose and zero fill only. | 500 epochs, dual prediction from epoch 100, evaluation every 100 epochs, final output. Fixed model seed 4; mask seed varies. |
| [DCP](https://github.com/XLearning-SCU/2022-TPAMI-DCP) | Native two-view or three-view complete-graph variant; first two Handwritten views. Native loader preprocessing and zero filling. | 100 epochs, batch 256, hidden dimension 1024, evaluation interval 100; final epoch. |
| [ICMVC](https://github.com/liunian-Jay/ICMVC) | First two views; feature standardization before masking; missing-aware top-10 adjacency. CPU route with optional-dependency compatibility adjustments. | 100 epochs, batch 256, hidden/latent dimensions 512/128, learning rate 1e-3; final epoch. |
| [GHICMC](https://github.com/KelvinXuu/GHICMC) | All views; feature min–max scaling before masking; observed-sample top-10 graph expanded to sample indices. | 30 epochs, hidden/latent dimensions 512/128, learning rate 1e-3; final epoch. |
| [HSACC](https://github.com/XiaojianDing/2025-NeurIPS-HSACC) | First two Fashion views; all six Handwritten views. Handwritten min–max scaling before masking; two-view loader values retained. | 120 epochs, batch 256, learning rate 1e-4; final epoch, excluding native best-by-label metrics. |
| [RAC-DMVC](https://github.com/LouisDong95/RAC-DMVC) | First two views; standardization before masking and incomplete-sample adaptation. | 30 epochs including 10 warm-up epochs, batch 512, hidden/embedding dimensions 512/128, learning rate 5e-4; final output. |
| [LSIMVC](https://github.com/justsmart/LSIMVC) | All views; author normalization and graph routines in MATLAB R2025a. Rectangular-polar initialization for Handwritten, where a view dimension is below K. | k=5, lambda=1, beta=0.01, r=2; 100-update cap, tolerance 1e-6. Method seed 1; graph and final clustering use the protocol seed. Final K-means: 20 restarts, 100 iterations maximum. |
| [PSIMVC-PG](https://github.com/wangsiwei2010/PSIMVC-PG) | All views; zero missing rows before author sample-vector normalization. | Latent dimension and anchor count K; at most 51 updates or objective/relative tolerance 1e-3. Final K-means: 20 restarts, 1,000 iterations maximum. |

DCP, ICMVC, GHICMC, HSACC and RAC-DMVC initialize each run with the protocol seed; PSIMVC-PG initializes its author routine with that seed. Public links identify the author releases. Exact historical upstream revisions and complete tuning histories of the six deep baselines are not available. The recorded method-specific budgets and preprocessing are retained.

## IMSC-AGL and RISE (Supplementary Section S1.2)

Both comparisons use BDGP and Handwritten, all views, the same five fractions and seeds 1–3. Final MATLAB K-means uses the protocol seed, 20 restarts and at most 100 iterations. Predictions are fixed before independent ACC, arithmetic-normalized NMI and ARI evaluation.

- **IMSC-AGL:** author archive linked from [the publication page](https://sites.google.com/view/jerry-wen-hit/publications), under *Incomplete Multiview Spectral Clustering with Adaptive Graph Learning*. Available sample rows are normalized to unit length. The implementation retains binary three-neighbor graphs, complete eigendecomposition/SVD, author initialization/update order, and BBCSport-demo settings: lambda1=0.01, lambda2=1e5, lambda3=0.1, initial mu=0.01, multiplier=1.1. Stop at 100 updates or absolute successive-objective change below 1e-7 after iteration 2. Row-normalize the final common embedding before clustering.
- **RISE:** [author repository](https://github.com/W-Xinxin/AAAI2025_RISE), commit `d66c0bf5ff9787aa9df028801010e519178fe6e2`. Use 4K anchors, embedding dimension 2K, five neighbors and beta=100. Retain author observed-sample extraction, normalization, anchor construction (seed 6666, ten internal clustering restarts), bipartite graph and alternating SVD updates. Obtain sample count from mask dimensions. Use the final embedding without extra row normalization. Stop at absolute successive-objective change below 1e-3 after iteration 19, with a 100-update cap.

## Additional studies (Supplementary Section S4)

Both studies use BDGP at eta=0.5 and Handwritten at eta=0.7, with full views, one-view-drop masks and seeds 1–3.

**Training duration:** train one continuous mapper trajectory from initialization to epoch 240 with a fixed graph and target, retaining budgets 60, 120 and 240. Keep the optimizer state across updates. Apply the unchanged label-free selector separately to checkpoints within each budget; evaluate every five epochs with dropout disabled. Architecture, losses, initialization, optimizer and learning rate follow the standard configuration; omit the optional corrector. This extended procedure is separate from the fixed 120-epoch command provided here.

**K versus K+1 row scaling:** obtain a common ordered, signed K+1-dimensional basis with one eigensolver call. Compare row-normalizing all K+1 coordinates before retaining the first K with retaining the first K coordinates before row normalization. Both outputs have K columns and use the same K-means seed and 20 restarts. This study evaluates graph partitions without mapper or corrector fitting.

## Metrics and aggregation

ACC uses optimal assignment of cluster IDs; NMI uses arithmetic normalization; ARI is the adjusted Rand index. Cell means average three seeds and dataset means average five fractions. SD is the sample standard deviation. Paired summaries first average the five within-seed ROSE-minus-comparator differences, then summarize the three seed averages. Score differences are expressed in percentage points. FV and RV results remain separate.
