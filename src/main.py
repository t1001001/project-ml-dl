import os
import utils.config as conf
from utils.cleanup import cleanup
from data_prep.get_dataset import get_dataset
from data_prep.split_dataset import split_dataset
from data_prep.data_augmentation import process_images

TRAIN_FOLDERS = [
    conf.DATASET_TRAIN_BUTTERFLY_PATH,
    conf.DATASET_TRAIN_CAT_PATH,
    conf.DATASET_TRAIN_CHICKEN_PATH,
    conf.DATASET_TRAIN_COW_PATH,
    conf.DATASET_TRAIN_DOG_PATH,
    conf.DATASET_TRAIN_ELEPHANT_PATH,
    conf.DATASET_TRAIN_HORSE_PATH,
    conf.DATASET_TRAIN_SHEEP_PATH,
    conf.DATASET_TRAIN_SPIDER_PATH,
    conf.DATASET_TRAIN_SQUIRREL_PATH,
]
TEST_FOLDERS = [
    conf.DATASET_TEST_BUTTERFLY_PATH,
    conf.DATASET_TEST_CAT_PATH,
    conf.DATASET_TEST_CHICKEN_PATH,
    conf.DATASET_TEST_COW_PATH,
    conf.DATASET_TEST_DOG_PATH,
    conf.DATASET_TEST_ELEPHANT_PATH,
    conf.DATASET_TEST_HORSE_PATH,
    conf.DATASET_TEST_SHEEP_PATH,
    conf.DATASET_TEST_SPIDER_PATH,
    conf.DATASET_TEST_SQUIRREL_PATH,
]

def main():
    print("Running the project!")
    get_dataset()
    split_dataset()
    for folder in TRAIN_FOLDERS:
        process_images(folder, apply_augmentation=True)
    for folder in TEST_FOLDERS:
        process_images(folder, apply_augmentation=True)
    cleanup()
    print("Finished running the project!")

if __name__ == "__main__":
    main()