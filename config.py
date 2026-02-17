from pathlib import Path
from typing import Literal, TypeAlias


DeviceType: TypeAlias = Literal["cuda", "cpu"]


# Training settings
SCALING_FACTOR = 4

# Dataset pathes
PRETRAIN_DATASET_PATH = Path("data/ImageNet.txt")
TRAIN_DATASET_PATH = Path("data/DF2K")
VAL_DATASET_PATH = Path("data/DIV2K_val")
TEST_DATASET_PATHS = [
    Path("data/Set5"),
    Path("data/Set14"),
    Path("data/BSDS100"),
    Path("data/Urban100"),
    Path("data/Manga109"),
]
