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
    print("Running the project!")
    get_dataset()
    print("Finished getting the dataset!")
    split_dataset()
    print("Finished splitting the dataset!")
    for folder in TRAIN_FOLDERS:
        process_images(folder, apply_augmentation=True)
    print("Finished augmenting the dataset!")
    cleanup()
    print("Cleaned up old ressources!")
    train_alexnet()
    print("Finished training AlexNet!")
    train_mobilenetv2()
    print("Finished training MobileNetV2!")
    train_resnet50()
    print("Finished training ResNet50")
    evaluate_alexnet()
    print("Finished evaluating AlexNet!")
    evaluate_mobilenetv2()
    print("Finished evaluating MobileNetV2!")
    evaluate_resnet50()
    print("Finished evaluating ResNet50!")
    print("Finished running the project!")

if __name__ == "__main__":
    main()