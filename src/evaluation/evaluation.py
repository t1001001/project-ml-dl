from evaluation.eval_alexnet import _evaluate_alexnet
from evaluation.eval_mobilenetv2 import _evaluate_mobilenetv2
from evaluation.eval_resnet50 import _evaluate_resnet50

def evaluate():
    """
    Evaluates all the models.
    """
    print("--- Starting model evaluation ---")
    _evaluate_alexnet()
    _evaluate_mobilenetv2()
    _evaluate_resnet50()
    print("--- Finished model evaluation ---")

if __name__ == "__main__":
    evaluate()