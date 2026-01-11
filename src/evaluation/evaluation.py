from evaluation.eval_custom import _evaluate_custom
from evaluation.eval_mobilenetv2 import _evaluate_mobilenetv2
from evaluation.eval_resnet50 import _evaluate_resnet50
from evaluation.visualize_results import visualize_all

def evaluate():
    print("=" * 50)
    print("STARTING MODEL EVALUATION")
    print("=" * 50)
    
    results = []
    results.append(_evaluate_custom())
    results.append(_evaluate_mobilenetv2())
    results.append(_evaluate_resnet50())
    
    print("\n" + "=" * 50)
    print("GENERATING VISUALIZATIONS")
    print("=" * 50)
    
    visualize_all(results)
    
    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    evaluate()