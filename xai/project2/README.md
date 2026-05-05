# XAI Project 2: Representation Analysis of Multi-Task Vision Transformers

Internal-representation analysis of the three Vision Transformer variants (classifier-only, saliency-only, multi-task) trained in Project 1 (`~/projects/xai/project1/`). Project 1 evaluated five output-level explanation methods and showed that attribution maps differ across the three variants. Project 2 asks whether those output-level differences are anchored in genuinely different internal representations, or whether the models encode the input similarly and the disagreement originates in the explanation methods themselves.

This project is for MTH 4326 (Explainable AI) at Florida Institute of Technology.

## Methods

| Method | Family | Idea |
|---|---|---|
| Linear probing | Decodability | Per-layer linear probes for class label, saliency, and object location, with random-label control. |
| Centered Kernel Alignment | Representation similarity | Pairwise similarity between layer activations across the three variants, debiased estimator. |
| Procrustes / SVCCA | Representation similarity | Complementary metrics, since CKA is sensitive to subset translations and population structure. |
| Patch-token clustering | Geometry | K-means on patch-token embeddings, with cluster purity and NMI against COCO labels and saliency quartiles. |
| Sparse autoencoder (exploratory) | Feature decomposition | TopK SAE on the multi-task variant's mid-layer residual stream, with feature ablation against both heads. |
| Grad-CAM (class-method baseline, Lecture 9) | Attribution | Representation-space attribution at block 10 (Selvaraju et al.\ 2017). The class-method requirement for Project 2 benchmarking. |
| SHAP (additional baseline) | Attribution | Kernel SHAP at the 14×14 patch grid, included as a perturbation-based comparison to Grad-CAM. |

## Requirements

- Python 3.8+
- CUDA-capable GPU (~6 GB for SAE training, less for the rest)

```bash
pip install -r requirements.txt
```

## Data and Models

This project reuses the trained checkpoints and datasets from Project 1. `src/config.py` points at `~/projects/xai/project1/runs/best_{multitask,cls_only,sal_only}.pt` and `~/projects/xai/project1/datasets/{salicon,coco2014}/`.

If those paths are missing, retrain Project 1 first:

```bash
cd ~/projects/xai/project1
python -m src.train.train_multitask --mode multitask
python -m src.train.train_multitask --mode cls_only
python -m src.train.train_multitask --mode sal_only
```

## Pipeline

All commands run from the project2 root.

### 1. Linear probing across all 12 layers

```bash
python -m src.representation.probes
```

Per-layer linear probes for class label (multi-label, mAP), saliency at low resolution (RidgeCV, R²), and quadrant-of-attention (multinomial logistic, accuracy). Random-label control probe per layer.

Saves `runs/probes.json`.

### 2. CKA, Procrustes, SVCCA across variants

```bash
python -m src.representation.cka
```

Pairwise debiased CKA (linear and RBF), Procrustes alignment, and SVCCA (top-20 components) between every pair of variants at each layer.

Saves `runs/cka.json`.

### 3. Patch-token clustering

```bash
python -m src.representation.clustering
```

K-means on final-layer patch tokens. Reports cluster purity and normalized mutual information against the dominant COCO category and against per-image saliency quartiles. Includes seed-stability ARI.

Saves `runs/clustering.json`.

### 4. Sparse autoencoder feature decomposition (exploratory)

```bash
python -m src.representation.sae
```

Trains a TopK SAE (dictionary 4096, k=32) on patch-token activations from block 6 of the multi-task variant. Ablates the top features one at a time and measures the change in classification and saliency outputs (with an SAE-reconstruction-only baseline as the reference).

Saves `runs/sae_results.json` and `runs/sae_multitask_layer6.pt`.

### 5. Grad-CAM class-method baseline (Lecture 9)

```bash
python -m src.xai.grad_cam
```

Grad-CAM at block 10 (second-to-last). Evaluates deletion/insertion AUC and human alignment.

Saves `runs/grad_cam_results.json`.

### 5b. SHAP additional baseline

```bash
python -m src.xai.shap_explain
```

Kernel SHAP at the 14×14 patch grid with 200 mask samples per image.

Saves `runs/shap_results.json`.

### 6. Generate figures

```bash
python -m src.utils.generate_figures
```

Writes the four main report figures into `runs/figures/`.

### 7. Print consolidated table

```bash
python -m src.utils.results_table
```

## Configuration

All hyperparameters live in [src/config.py](src/config.py).

| Parameter | Value | Description |
|---|---|---|
| `ACTIVATION_SAMPLES` | 1000 | Val-set images for representation analysis |
| `SAE_LAYER` | 6 | Mid-layer block index for SAE training |
| `SAE_DICT_SIZE` | 4096 | SAE dictionary width |
| `SAE_TOPK` | 32 | TopK sparsity |
| `SAE_EPOCHS` | 20 | SAE training epochs |
| `SEED` | 42 | Reproducibility seed |

## Project Structure

```
project2/
  src/
    config.py
    seed.py
    data/
      salicon_coco.py     # copied from project 1
      transforms.py       # copied from project 1
    models/
      vit_multitask.py    # copied from project 1
    representation/
      activations.py      # forward-hook collection of residual-stream activations
      probes.py           # per-layer linear probes + random-label control
      cka.py              # debiased CKA, Procrustes, SVCCA
      clustering.py       # patch-token clustering with purity / NMI
      sae.py              # TopK SAE + feature ablation
    xai/
      shap_explain.py     # kernel SHAP baseline
    utils/
      load_models.py      # load all three variants
      io.py               # checkpoint i/o (copied from project 1)
      metrics.py          # mAP / CC / SIM / KL (copied from project 1)
      generate_figures.py # all report figures
      results_table.py    # consolidated table
  notebooks/
  report/
    xai/                  # MTH 4326 report (NeurIPS format)
    powerpoints/
      xai/                # Beamer slides
  runs/                   # results, checkpoints, figures (not tracked)
  brainstorming/
  requirements.txt
```

## Reports

| Course | Report | Focus |
|---|---|---|
| MTH 4326 (Explainable AI) | `report/xai/main.tex` | Probing, CKA / Procrustes / SVCCA, clustering, SAE feature decomposition |

Presentation: `report/powerpoints/xai/main.tex` (Beamer).
