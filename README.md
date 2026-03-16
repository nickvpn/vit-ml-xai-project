# Multi-Task Vision Transformer with XAI Benchmarking

A multi-task Vision Transformer (DeiT-Small) that jointly performs multi-label image classification (COCO categories) and human saliency prediction (SALICON), with a full explainability analysis pipeline.

## Requirements

- Python 3.8+
- CUDA-capable GPU (training assumes `cuda`)
- ~4 GB GPU memory (DeiT-Small with batch size 32)

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: torch, torchvision, timm, numpy, scipy, scikit-learn, matplotlib, Pillow, lime

## Dataset Preparation

The project expects two datasets placed under `datasets/`:

```
datasets/
  salicon/
    images/
      train/          # 10,000 COCO-format JPGs (COCO_train2014_XXXXXXXXXXXX.jpg)
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

## Running the Full Pipeline

All commands are run from the project root directory.

### Step 1: Train the models

```bash
# multitask model (classification + saliency jointly)
python -m src.train.train_multitask --mode multitask

# ablation variants
python -m src.train.train_multitask --mode cls_only
python -m src.train.train_multitask --mode sal_only
```

Each training run:
- Fine-tunes DeiT-Small for 15 epochs with cosine annealing (lr=1e-4)
- Saves best checkpoint (by val loss) to `runs/best_{mode}.pt`
- Saves final checkpoint to `runs/final_{mode}.pt`
- Saves training history to `runs/history_{mode}.json`
- Prints val mAP and loss each epoch

Expect ~5-10 minutes per variant on a modern GPU.

### Step 2: Run baselines

```bash
python -m src.train.baselines
```

Extracts frozen DeiT features (CLS token, 384-dim) and trains:
- **Classification**: Logistic Regression (OVR), MLP per-label
- **Saliency**: Mean saliency map, Ridge regression, MLP regressor

Results saved to `runs/baseline_results.json`.

### Step 3: Evaluate trained models

```bash
python -m src.train.evaluate --mode multitask
python -m src.train.evaluate --mode cls_only
python -m src.train.evaluate --mode sal_only
```

Computes mAP, CC, SIM, and KL-div on the validation set. Results saved to `runs/eval_{mode}.json`.

### Step 4: Run XAI analysis

```bash
python -m src.xai.run_xai --n_samples 100
```

For each sample, runs five explanation methods and evaluates them:

| Method | What it does |
|---|---|
| Gradient saliency | Absolute value of input gradients |
| Grad x input | Element-wise gradient times input |
| Integrated gradients | Accumulated gradients along interpolation path (50 steps) |
| Attention rollout | Multiplied attention matrices across all transformer layers |
| LIME | Local surrogate model with 14x14 patch-grid segmentation |

Evaluation metrics computed per method:
- **Faithfulness**: deletion AUC (lower = better), insertion AUC (higher = better)
- **Stability**: cosine similarity under Gaussian noise, horizontal flip, brightness shift
- **Human alignment**: CC and SIM against SALICON ground truth saliency

Results saved to `runs/xai_results.json`. Comparison figures for the first 5 samples saved to `runs/figures/`.

### Step 5: Generate report figures

```bash
python -m src.utils.generate_figures
```

Generates all summary plots in `runs/figures/`:
- `training_curves_{mode}.png` — loss and mAP over epochs
- `baseline_comparison.png` — bar charts comparing baselines vs fine-tuned models
- `xai_faithfulness_alignment.png` — deletion/insertion AUC and human alignment by method
- `xai_stability.png` — perturbation stability by method

### Step 6: Print consolidated results

```bash
python -m src.utils.results_table
```

Prints a formatted table of all results (baselines, model eval, XAI metrics).

## Notebooks

Interactive exploration is available in `notebooks/`:

| Notebook | Purpose |
|---|---|
| `01_data_and_model.ipynb` | Data sanity checks, sample visualization, model inference demo |
| `02_xai_analysis.ipynb` | Side-by-side explanation comparison, XAI results summary |

Run with Jupyter from the project root so imports resolve correctly.

## Configuration

All hyperparameters and paths are in [src/config.py](src/config.py):

| Parameter | Value | Description |
|---|---|---|
| `VIT_MODEL_NAME` | `deit_small_patch16_224` | Backbone model (from timm) |
| `NUM_LABELS` | 20 | Top-K COCO categories for classification |
| `BATCH_SIZE` | 32 | Training and eval batch size |
| `NUM_EPOCHS` | 15 | Training epochs |
| `LR` | 1e-4 | Learning rate |
| `SAL_LOSS_WEIGHT` | 1.0 | Weight of saliency loss in multitask mode |
| `IMG_SIZE` | 224 | Input image resolution |
| `GRID_SIZE` | 14 | Saliency map resolution (matches ViT patch grid) |

## Project Structure

```
project1/
  src/
    config.py                  # paths, hyperparameters, device
    seed.py                    # reproducibility (seed=42)
    data/
      salicon_coco.py          # dataset: pairs SALICON images with COCO labels
      transforms.py            # image transforms (ImageNet norm) + saliency transforms
    models/
      vit_multitask.py         # DeiT-Small + cls_head + sal_head, build_model() factory
    train/
      train_multitask.py       # training loop (multitask / cls_only / sal_only)
      baselines.py             # sklearn baselines on frozen DeiT features
      evaluate.py              # compute mAP, CC, SIM, KL on val set
    xai/
      gradient_saliency.py     # vanilla gradient + grad x input
      ig_explain.py            # integrated gradients
      lime_explain.py          # LIME with patch-grid segmentation
      attention_rollout.py     # attention rollout across transformer layers
      faithfulness.py          # deletion and insertion AUC tests
      stability.py             # perturbation stability (noise, flip, brightness)
      human_alignment.py       # CC/SIM against human saliency ground truth
      run_xai.py               # orchestrates full XAI analysis
    utils/
      metrics.py               # mAP, CC, SIM, KL-div
      viz.py                   # plotting and visualization helpers
      io.py                    # checkpoint save/load
      generate_figures.py      # generates all report figures
      results_table.py         # prints consolidated results table
  notebooks/
    01_data_and_model.ipynb    # data exploration + model demo
    02_xai_analysis.ipynb      # XAI visualization + results
  datasets/                    # SALICON + COCO data (not tracked)
  runs/                        # checkpoints, results JSONs, figures (not tracked)
  requirements.txt
```

## Quick Start (TL;DR)

```bash
# install
pip install -r requirements.txt

# train all variants
python -m src.train.train_multitask --mode multitask
python -m src.train.train_multitask --mode cls_only
python -m src.train.train_multitask --mode sal_only

# baselines + evaluation
python -m src.train.baselines
python -m src.train.evaluate --mode multitask
python -m src.train.evaluate --mode cls_only
python -m src.train.evaluate --mode sal_only

# xai analysis + figures
python -m src.xai.run_xai --n_samples 100
python -m src.utils.generate_figures
python -m src.utils.results_table
```
