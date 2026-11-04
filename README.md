# ROSE

Reference implementation of **ROSE: Reliability-Aware Observed-Support Spectral Embedding for Incomplete Multi-View Clustering**.

ROSE constructs a graph on co-observed neighborhoods, adjusts positive weights on that support, and transfers its fixed spectral target to a mask-aware mapper. An optional bounded corrector is applied after mapper selection.

## Installation

The recorded environment is Python 3.13.5, NumPy 2.4.3, SciPy 1.17.1, scikit-learn 1.8.0 and PyTorch 2.11.0+cu126. The reference configuration executes on CPU with four PyTorch threads; a GPU is not required for the commands below. The requirements retain the recorded PyTorch build.

Run these commands from the repository root in a Python 3.13 environment:

```bash
python -m pip install -r requirements.txt
```

## Data and missing-view masks

Obtain the five processed benchmark matrices from the public sources listed in [data_sources.csv](data_sources.csv). Keep the sample order and view order unchanged. The CSV gives filenames, shapes, feature and label fields, and checksums identifying the evaluated matrices. Store them in a local directory, then run:

```bash
python scripts/validate_data.py --data-dir /path/to/datasets
python scripts/generate_masks.py
```

The second command generates 75 masks in `masks/one_view_drop/` and checks each against [mask_manifest.csv](masks/mask_manifest.csv). It uses only sample counts, view counts, incomplete-sample fractions and seeds. Exactly `floor(eta*n)` samples each lose one randomly selected view. The settings are five datasets, `eta = 0.1, 0.3, 0.5, 0.7, 0.9`, and seeds `1, 2, 3`.

## Run ROSE

```bash
python run_rose.py --dataset bdgp --data /path/to/datasets/BDGP.mat --eta 0.5 --seed 1 --output-dir outputs/bdgp_eta0.50_seed1
```

Dataset identifiers are `bdgp`, `fashion`, `handwritten`, `hdigit` and `mnist_usps`. Their cluster counts are supplied in [config/frozen_method.json](config/frozen_method.json). Use a different output directory for each setting.

The command constructs the graph, trains and selects the mapper, applies the label-free correction decision, and saves the final partition and run summary. To compute final ACC, NMI, ARI and purity, add `--evaluate`. Ground-truth labels are loaded only after the partition and all selection decisions are fixed.

```bash
python run_rose.py --dataset bdgp --data /path/to/datasets/BDGP.mat --eta 0.5 --seed 1 --output-dir outputs/bdgp_eta0.50_seed1 --evaluate
```

## Implementation and experimental settings

- [Implementation notes](docs/IMPLEMENTATION.md): network operations, spectral normalization, checkpoint selection and bounded correction; corresponds to main-text Sections 3–5, Table 3 and Supplementary Section S1.3.
- [Experimental protocol](docs/EXPERIMENTAL_PROTOCOL.md): data handling, baseline sources and settings, and the additional studies described in Supplementary Sections S1 and S4.
- `rose/`: ROSE implementation. The mapper schedule is fixed at 120 epochs and the optional corrector at 40 epochs.
- `tests/`: small checks for mask generation, checkpoint selection and the effective-rank denominator. Run them without benchmark data or model fitting:

```bash
python -m unittest discover -s tests -v
```

The code update uses `e(U0)` in the effective-rank ratio, retains epoch 0, and searches all recorded checkpoints when no valid candidate exists. The complete rules are in the implementation notes.

This repository contains source code, configuration and input metadata. Dataset matrices, generated masks, fitted weights and experimental output files are obtained or generated locally. Third-party baselines are accessed through their author repositories; their source code is not bundled here. The baseline settings and additional-study procedures are documented, while the executable entry point above runs the standard ROSE configuration.

## Citation

See [CITATION.cff](CITATION.cff) for the manuscript title and authors.
