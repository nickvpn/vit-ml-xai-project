from pathlib import Path

# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
P1_ROOT = PROJECT_ROOT.parent / "project1"
DATA_DIR = P1_ROOT / "datasets"
SALICON_DIR = DATA_DIR / "salicon"
COCO_ANN_DIR = DATA_DIR / "coco2014" / "annotations"
P1_RUNS_DIR = P1_ROOT / "runs"
RUNS_DIR = PROJECT_ROOT / "runs"

# images and saliency, matched to project 1
IMG_SIZE = 224
GRID_SIZE = 14
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# model
VIT_MODEL_NAME = "deit_small_patch16_224"
NUM_LABELS = 20
EMBED_DIM = 384
NUM_BLOCKS = 12

# checkpoints from project 1
MODES = ["multitask", "cls_only", "sal_only"]
CKPT_PATHS = {m: P1_RUNS_DIR / f"best_{m}.pt" for m in MODES}

# analysis defaults
ANALYSIS_BATCH_SIZE = 32
ACTIVATION_SAMPLES = 1000  # number of val images for representation analysis
PROBE_BATCH = 256
SAE_LAYER = 6  # mid-layer for sae training
SAE_DICT_SIZE = 4096
SAE_TOPK = 32
SAE_EPOCHS = 20
SAE_LR = 1e-3
SAE_BATCH = 4096

# device
DEVICE = "cuda"
SEED = 42
