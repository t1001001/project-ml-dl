from models.build_alexnet import _train_alexnet
from models.build_mobilenetv2 import _train_mobilenetv2
from models.build_resnet50 import _train_resnet50

def build():
    """
    Builds and trains all models.
    """
    print("--- Starting model building ---")
    _train_alexnet()
    _train_mobilenetv2()
    _train_resnet50()
    print("--- Finished model building ---")

if __name__ == "__main__":
    build()