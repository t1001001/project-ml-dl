import os
import utils.config as conf
from utils.cleanup import cleanup
from data_prep.get_dataset import get_dataset
from data_prep.split_dataset import split_dataset
from data_prep.data_augmentation import process_images
from models.alexnet import train_alexnet

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
    print("Running the project!")
    print("Getting the dataset now!")
    get_dataset()
    print("Got the dataset - will spilit it now!")
    split_dataset()
    print("Splitted the dataset - will augment it now!")
    for folder in TRAIN_FOLDERS:
        process_images(folder, apply_augmentation=True)
    print("Augmented the dataset - will clean up old ressources!")
    cleanup()
    print("Cleaned up old ressources - will train an AlexNet model now!")
    train_alexnet()
    print("Finished running the project!")

if __name__ == "__main__":
    main()