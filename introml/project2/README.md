# Intro to ML Project 2: Random Features, Kernel Approximation, and Double Descent

Empirical study of random-feature ridge regression on a tabular regression task. The setup connects three foundational ideas: random features as a kernel approximation, the double-descent test-error curve, and the three-factor explanation of double descent from Schaeffer et al.\ (2023).

This project is for CSE 4224 (Intro to Machine Learning) at Florida Institute of Technology.

## Overview

We use random Fourier features (Rahimi and Recht, 2007) to approximate an RBF kernel, then sweep the number of features `P` from underparameterized through the interpolation threshold (`P = N`) to overparameterized. We reproduce the double-descent test-error spike and run three ablations that each remove one of the factors hypothesized to drive the spike:

1. **Ridge regularization.** Increase `λ` to suppress small singular values of the random feature matrix.
2. **Leading-mode projection.** Project test features onto the top-`k` singular directions of the training feature matrix.
3. **Noiseless target.** Replace `y` with the best linear fit on raw features (a noise-free target).

Each ablation is predicted to flatten the spike. We report kernel-PCA-style spectral analysis to confirm the singular-value collapse near `P = N`.

## Requirements

- Python 3.8+
- numpy, scipy, scikit-learn, matplotlib

```bash
pip install -r requirements.txt
```

## Dataset

UCI California Housing (8 features, ~20{,}000 rows). Loaded via scikit-learn's built-in loader. Standardized features and target. Training subset of size `N = 256` to make the interpolation threshold cheap to span.

## Pipeline

All commands run from the project2 root.

```bash
# 1. classical baselines: linear, ridge, mean predictor, exact RBF kernel ridge
python -m src.models.baselines

# 2. random-feature ridge sweep (the main experiment)
python -m src.models.random_feature_ridge

# 3. three-factor ablation
python -m src.models.three_factor_ablation

# 4. kernel-pca and spectral analysis
python -m src.models.kernel_pca_spectral

# 5. figures
python -m src.utils.generate_figures

# 6. consolidated table
python -m src.utils.results_table
```

## Configuration

All hyperparameters live in [src/config.py](src/config.py).

| Parameter | Value | Description |
|---|---|---|
| `N_TRAIN` | 256 | Training subset size, defines `P = N` |
| `P_VALUES` | `[4, 8, ..., 4096]` | Random-feature counts to sweep |
| `N_FEATURE_SEEDS` | 3 | Replicate seeds per `P` |
| `DEFAULT_RIDGE` | 1e-3 | Default ridge `λ` for the main sweep |
| `KPCA_COMPONENTS` | 32 | Kernel-PCA component count |
| `SEED` | 42 | Reproducibility seed |

## Project Structure

```
project2/
  src/
    config.py
    seed.py
    data/
      load.py                  # california housing loader, splits, standardization
    features/
      random_features.py       # rahimi-recht cosine features, median-heuristic bandwidth
    models/
      baselines.py             # linear, ridge, mean, exact rbf-krr
      random_feature_ridge.py  # main sweep
      three_factor_ablation.py # schaeffer ablations
      kernel_pca_spectral.py   # kpca + spectrum at varying P
    utils/
      generate_figures.py      # all report figures
      results_table.py         # consolidated table
  notebooks/
  report/
    intro_to_ml/               # CSE 4224 report (NeurIPS format)
    powerpoints/
      intro_to_ml/             # Beamer slides
  runs/                        # results, figures (not tracked)
  requirements.txt
```

## Reports

| Course | Report | Focus |
|---|---|---|
| CSE 4224 (Intro to ML) | `report/intro_to_ml/main.tex` | Random features, double descent, three-factor ablations |

Presentation: `report/powerpoints/intro_to_ml/main.tex` (Beamer).
