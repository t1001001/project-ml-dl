from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import utils.config as conf
import time

def _evaluate_mobilenetv2(image_size=(224, 224), batch_size=64):
    print("\n--- Evaluating MobileNetV2 ---")
    model = load_model(conf.SAVED_MODELS_PATH+"/mobilenetv2.h5")
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        conf.DATASET_TEST,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        label_mode='int'
    )
    class_names = test_ds.class_names
    test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    num_samples = sum(1 for _ in test_ds.unbatch())
    start_time = time.time()
    predictions = model.predict(test_ds)
    inference_time = time.time() - start_time
    avg_inference_time_ms = (inference_time / num_samples) * 1000
    
    predicted_classes = np.argmax(predictions, axis=1)
    true_labels = np.concatenate([y for x, y in test_ds], axis=0)
    
    cm = confusion_matrix(true_labels, predicted_classes)
    acc = accuracy_score(true_labels, predicted_classes)
    f1_per_class = f1_score(true_labels, predicted_classes, average=None)
    precision_per_class = precision_score(true_labels, predicted_classes, average=None)
    recall_per_class = recall_score(true_labels, predicted_classes, average=None)
    f1_macro = f1_score(true_labels, predicted_classes, average='macro')
    f1_weighted = f1_score(true_labels, predicted_classes, average='weighted')
    
    print(f"Accuracy: {acc:.2%}")
    print(f"Timing: {inference_time:.2f}s total, {avg_inference_time_ms:.2f}ms per image")
    
    return {
        'name': 'MobileNetV2',
        'accuracy': acc,
        'f1_scores': f1_per_class.tolist(),
        'precision_scores': precision_per_class.tolist(),
        'recall_scores': recall_per_class.tolist(),
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'inference_time_ms': avg_inference_time_ms,
        'class_names': class_names
    }

if __name__ == "__main__":
    _evaluate_mobilenetv2()