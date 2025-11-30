from utils.get_root_path import get_root_path
from typing import Final

ROOT: Final = get_root_path()

DATASET_PATH: Final = f"{ROOT}/dataset"

IMAGE_TRAIN_PATH: Final = f"{ROOT}/dataset/train"
IMAGE_TEST_PATH: Final = f"{ROOT}/dataset/test"