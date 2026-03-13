from pathlib import Path
from typing import Literal, TypeAlias

DeviceType: TypeAlias = Literal["cuda", "cpu"]

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg"]

# Architecture settings
NUM_CHANNELS = 180
NUM_RHAG_BLOCKS = 6
NUM_HAB_BLOCKS = 6
NUM_HEADS = 6
WINDOW_SIZE = 16

SQUEEZE_FACTOR = 30
COMPRESS_RATIO = 3
CAB_SCALE = 0.01
OVERLAP_RATIO = 0.5
MLP_RATIO = 2

# Training settings
SCALING_FACTOR = 4
PATCH_SIZE = 64

CURRENT_PHASE: Literal["pretraining", "fine-tuning"] = "fine-tuning"

if CURRENT_PHASE == "pretraining":
    # Pretraining settings
    EFFECTIVE_BATCH_SIZE = 32
    BATCH_SIZE = 8
    NUM_ITERATIONS = 100_000
    LEARNING_RATE = 2e-4

    SCHEDULER_MILESTONES = [37_500, 62_500, 81_250, 87_500, 93_750]
    SCHEDULER_GAMMA = 0.5


elif CURRENT_PHASE == "fine-tuning":
    # Fine-tuning settings
    EFFECTIVE_BATCH_SIZE = 32
    BATCH_SIZE = 8
    NUM_ITERATIONS = 50_000
    LEARNING_RATE = 1e-5

    SCHEDULER_MILESTONES = [25_000, 40_000, 45_000, 48_000]
    SCHEDULER_GAMMA = 0.5

GRADIENT_CLIPPING_NORM = 0.5
USE_GRADIENT_CHECKPOINTING = True

# Technical settings
LOG_FREQ = 10
VAL_FREQ = 1000
SAVE_CHECKPOINT_FREQ = 10000

# Optimizer settings
ADAM_BETAS = (0.9, 0.99)
ADAM_EPS = 1e-8

# Data loader settings
TRAIN_NUM_WORKERS = 8
TRAIN_PREFETCH_FACTOR = 4
VAL_NUM_WORKERS = 2
VAL_PREFETCH_FACTOR = 2

# Dataset pathes
PRETRAIN_DATASET_PATH = Path("data/ImageNet_filtered.txt")
TRAIN_DATASET_PATH = Path("data/DF2K")
VAL_DATASET_PATH = Path("data/Set14")
TEST_DATASET_PATHS = [
    Path("data/Set5"),
    Path("data/Set14"),
    Path("data/BSDS100"),
    Path("data/Urban100"),
    Path("data/Manga109"),
]

# Checkpoint settings
LOAD_BEST_CHECKPOINT = True
LOAD_CHECKPOINT = False

BEST_CHECKPOINT_DIR_PATH = Path("checkpoints/best")
CHECKPOINT_DIR_PATH = Path("checkpoints/iter_0")

# WanbB settings
USE_WANDB = False
WANDB_PROJECT_NAME = "HAT-SR"
WANDB_CONFIG = {
    "num_channels": NUM_CHANNELS,
    "num_rhag_blocks": NUM_RHAG_BLOCKS,
    "num_hab_blocks": NUM_HAB_BLOCKS,
    "num_heads": NUM_HEADS,
    "window_size": WINDOW_SIZE,
    "squeeze_factor": SQUEEZE_FACTOR,
    "compress_ratio": COMPRESS_RATIO,
    "cab_scale": CAB_SCALE,
    "overlap_ratio": OVERLAP_RATIO,
    "mlp_ratio": MLP_RATIO,
    "scaling_factor": SCALING_FACTOR,
    "patch_size": PATCH_SIZE,
    "batch_size": BATCH_SIZE,
    "num_iterations": NUM_ITERATIONS,
    "learning_rate": LEARNING_RATE,
    "gradient_clipping_norm": GRADIENT_CLIPPING_NORM,
    "use_gradient_checkpointing": USE_GRADIENT_CHECKPOINTING,
}
