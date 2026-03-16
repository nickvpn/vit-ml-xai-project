from pathlib import Path

# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "datasets"
SALICON_DIR = DATA_DIR / "salicon"
COCO_ANN_DIR = DATA_DIR / "coco2014" / "annotations"
RUNS_DIR = PROJECT_ROOT / "runs"

# images and saliency
IMG_SIZE = 224
GRID_SIZE = 14
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# model
VIT_MODEL_NAME = "deit_small_patch16_224"
NUM_LABELS = 20  # top-k coco categories

# training
BATCH_SIZE = 32
NUM_EPOCHS = 15
LR = 1e-4
WEIGHT_DECAY = 1e-4
SAL_LOSS_WEIGHT = 1.0  # lambda for multi-task loss
SAL_LOSS_TYPE = "kl"  # "mse" or "kl" for saliency loss

# device
DEVICE = "cuda"
