from utils.get_root_path import get_root_path
from typing import Final

ROOT: Final = get_root_path()

DATASET_PATH: Final = f"{ROOT}/datasets"

TRAIN_RATIO: Final = 0.8
TEST_RATIO: Final = 0.2