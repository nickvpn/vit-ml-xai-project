# Multi-Task Vision Transformer with XAI Benchmarking

Two course-project series at Florida Institute of Technology, sharing one multi-task Vision Transformer backbone (DeiT-Small) trained jointly on COCO classification and SALICON saliency.

| Project | CSE 4224 (Intro to ML) | MTH 4326 (Explainable AI) |
|---|---|---|
| **Project 1** | Multi-task learning, KL vs MSE loss, classical baselines | Five post-hoc XAI methods on faithfulness, stability, human alignment |
| **Project 2** | Random features, kernel approximation, double descent on tabular data | Internal-representation analysis (probes, CKA, clustering, sparse autoencoder) of the same three Project 1 ViT variants |

The repo is organized so Project 1 lives at the top level (the original code) and each Project 2 has its own subdirectory.

```
vit-ml-xai-project/
├── src/                      # Project 1 code (multi-task ViT + 5 XAI methods)
├── runs/                     # Project 1 checkpoints and figures
├── notebooks/                # Project 1 notebooks
├── scripts/                  # Project 1 figure helpers
├── report/                   # Project 1 reports (gitignored, kept locally)
├── requirements.txt          # Project 1 dependencies
│
├── xai/project2/             # Project 2: MTH 4326 representation analysis
│   ├── src/                  # probes, CKA, clustering, SAE, Grad-CAM, SHAP
│   ├── runs/                 # results JSONs and figures
│   ├── report/               # NeurIPS-format report and Beamer slides
│   ├── README.md
│   └── requirements.txt
│
└── introml/project2/         # Project 2: CSE 4224 random features and double descent
    ├── src/                  # random feature ridge, three-factor ablation, kernel PCA
    ├── runs/                 # results JSONs and figures
    ├── report/               # NeurIPS-format report and Beamer slides
    ├── README.md
    └── requirements.txt
```

---

## Project 1: Multi-Task ViT + XAI Benchmarking

A multi-task Vision Transformer (DeiT-Small) that jointly performs multi-label image classification and human saliency prediction, with a full explainability analysis pipeline.

### For the ML audience

We fine-tune a pretrained DeiT-Small to classify images into 20 COCO object categories and predict where humans look (saliency) at the same time. The key finding is that switching the saliency loss from MSE to KL divergence dramatically improves saliency prediction (CC: 0.196 to 0.899) with minimal impact on classification accuracy.

We also compare the fine-tuned model against classical baselines (logistic regression, MLP, ridge regression) that use frozen features from the same backbone, showing that end-to-end fine-tuning is necessary for strong saliency prediction.

### For the XAI audience

We apply five explanation methods to the trained classifier and evaluate them on three axes:

| Method | Family | Key idea |
|---|---|---|
| Gradient saliency | Gradient | How sensitive is the output to each pixel? |
| Gradient x input | Gradient | Gradient weighted by input intensity |
| Integrated gradients | Gradient (path) | Accumulated gradients from a zero baseline |
| Attention rollout | Architecture | Product of attention matrices across layers |
| LIME | Perturbation | Local linear surrogate on masked inputs |

The central finding is that faithfulness and human alignment are distinct (even inversely related). LIME is the most faithful but does not match human gaze. Integrated gradients offers the best balance across all three axes.

### Requirements

- Python 3.8+
- CUDA-capable GPU (training assumes `cuda`)
- ~4 GB GPU memory (DeiT-Small with batch size 32)

```bash
pip install -r requirements.txt
```

Dependencies: torch, torchvision, timm, numpy, scipy, scikit-learn, matplotlib, Pillow, lime.

### Dataset Preparation

The project expects two datasets placed under `datasets/` (gitignored):

```
datasets/
  salicon/
    images/
      train/          # 10,000 COCO-format JPGs
      val/            # 5,000 COCO-format JPGs
    train/            # 10,000 saliency map PNGs (matching image IDs)
    val/              # 5,000 saliency map PNGs
  coco2014/
    annotations/
      instances_train2014.json
      instances_val2014.json
```

- **SALICON**: download from [salicon.net](http://salicon.net/dataset/)
- **COCO 2014 annotations**: download from [cocodataset.org](https://cocodataset.org/#download)

The dataset class automatically pairs SALICON images with COCO category annotations by matching image IDs. It selects the top-20 most frequent COCO categories for multi-label classification.

### Pipeline

All commands run from the project root.

```bash
# train all three variants
python -m src.train.train_multitask --mode multitask
python -m src.train.train_multitask --mode cls_only
python -m src.train.train_multitask --mode sal_only

# classical baselines on frozen DeiT features
python -m src.train.baselines

# evaluate trained models
python -m src.train.evaluate --mode multitask
python -m src.train.evaluate --mode cls_only
python -m src.train.evaluate --mode sal_only

# XAI analysis (5 methods on 100 validation samples)
python -m src.xai.run_xai --n_samples 100

# figures and consolidated table
python -m src.utils.generate_figures
python -m src.utils.results_table
```

### Configuration

All hyperparameters live in [src/config.py](src/config.py).

| Parameter | Value | Description |
|---|---|---|
| `VIT_MODEL_NAME` | `deit_small_patch16_224` | Backbone (from timm) |
| `NUM_LABELS` | 20 | Top-K COCO categories |
| `BATCH_SIZE` | 32 | Training and eval batch size |
| `NUM_EPOCHS` | 15 | Training epochs |
| `LR` | 1e-4 | Learning rate |
| `SAL_LOSS_WEIGHT` | 1.0 | Lambda for saliency loss in multitask mode |
| `SAL_LOSS_TYPE` | `kl` | Saliency loss function (`kl` or `mse`) |
| `IMG_SIZE` | 224 | Input image resolution |
| `GRID_SIZE` | 14 | Saliency map resolution (matches patch grid) |

---

## Project 2 (XAI): Representation Analysis of Multi-Task Vision Transformers

Lives in [`xai/project2/`](xai/project2/). Internal-representation analysis of the three Project 1 ViT variants (classifier-only, saliency-only, multi-task), using methods entirely disjoint from Project 1's attribution methods.

**Question.** Project 1 found that the three variants produce different attribution maps. Are those differences anchored in real internal-representation differences, or are the explanation methods themselves noisy?

**Methods.** Per-layer linear probing, debiased Centered Kernel Alignment with Procrustes and SVCCA in parallel, patch-token clustering, and an exploratory TopK sparse autoencoder feature ablation. Grad-CAM and Kernel SHAP as attribution baselines.

**Headline finding.** The multi-task backbone is asymmetrically classifier-aligned. Procrustes alignment with classifier-only stays at 0.84 at the final layer while alignment with saliency-only falls to 0.41. Multi-task is the only variant whose patch-token clusters simultaneously carry both category and saliency structure.

```bash
cd xai/project2
pip install -r requirements.txt
python -m src.representation.probes
python -m src.representation.cka
python -m src.representation.clustering
python -m src.representation.sae
python -m src.xai.grad_cam
python -m src.xai.shap_explain
python -m src.utils.generate_figures
python -m src.utils.results_table
```

See [`xai/project2/README.md`](xai/project2/README.md) for full details.

---

## Project 2 (Intro ML): Random Features and Double Descent

Lives in [`introml/project2/`](introml/project2/). An empirical study of random-feature ridge regression on California Housing that connects three foundational ideas: kernel approximation by random Fourier features, the double-descent test-error spike, and the three-factor explanation of double descent from Schaeffer et al. 2023.

**Headline finding.** Test MSE peaks at $P \approx 300$ (just past $N = 256$) at 1.21, then descends to 0.75 at $P = 4096$. Each of the three Schaeffer factors (ridge regularization, leading-mode projection, noiseless target) eliminates the spike independently. The smallest singular value of the random feature matrix collapses to $6 \times 10^{-7}$ exactly at $P = N$, the structural witness for the spike.

```bash
cd introml/project2
pip install -r requirements.txt
python -m src.models.baselines
python -m src.models.random_feature_ridge
python -m src.models.three_factor_ablation
python -m src.models.kernel_pca_spectral
python -m src.utils.generate_figures
python -m src.utils.results_table
```

See [`introml/project2/README.md`](introml/project2/README.md) for full details.

---

## Reports

Project 1 reports live under `report/` (gitignored). Project 2 reports are tracked under each subdirectory's `report/`:

| Course | Project | Path |
|---|---|---|
| CSE 4224 (Intro to ML) | 1 | `report/intro_to_ml/main.tex` |
| MTH 4326 (Explainable AI) | 1 | `report/xai/main.tex` |
| CSE 4224 (Intro to ML) | 2 | `introml/project2/report/intro_to_ml/main.tex` |
| MTH 4326 (Explainable AI) | 2 | `xai/project2/report/xai/main.tex` |

Beamer slide decks are alongside each report under `report/powerpoints/`.

## Reproducibility

All experiments use seed 42. Project 2 reuses Project 1's trained checkpoints (`runs/best_*.pt`); no retraining is required for Project 2.
