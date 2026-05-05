from pathlib import Path

# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"

# dataset
DATASET = "california_housing"  # or "wine_quality"
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # of training portion
N_TRAIN = 256  # n_train used to define the interpolation threshold P=N

# random features
P_VALUES = [4, 8, 16, 32, 64, 128, 200, 240, 256, 272, 300, 350,
            512, 1024, 2048, 4096]
SIGMA = None  # None = median heuristic on training data
N_FEATURE_SEEDS = 3
RIDGE_LAMBDAS = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
DEFAULT_RIDGE = 1e-3

# kernel pca
KPCA_COMPONENTS = 32

# misc
SEED = 42
