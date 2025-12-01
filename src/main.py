import os
import utils.config as conf
from utils.cleanup import cleanup
from data_prep.get_dataset import get_dataset
from data_prep.split_dataset import split_dataset
from data_prep.data_augmentation import process_images
from models.alexnet import train_alexnet
from models.mobilenetv2 import train_mobilenetv2
from models.resnet50 import train_resnet50
from evaluation.alexnet_eval import evaluate_alexnet
from evaluation.mobilenetv2_eval import evaluate_mobilenetv2
from evaluation.resnet50_eval import evaluate_resnet50

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

def main():
    print("Getting the dataset!")
    get_dataset()
    split_dataset()
    for folder in TRAIN_FOLDERS:
        process_images(folder, apply_augmentation=True)
    cleanup()
    print("Finished getting the dataset!")

if __name__ == "__main__":
    main()