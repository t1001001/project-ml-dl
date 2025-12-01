from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import utils.config as conf
import os
import numpy as np

def evaluate_mobilenetv2(image_size=(224, 224), batch_size=64):
    """
    Evaluates the MobileNetV2 model.
    """
    model = load_model(conf.SAVED_MODELS_PATH+"/mobilenetv2.h5")
    test_dir = conf.DATASET_TEST
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        label_mode='int'
    )
    class_names = test_ds.class_names
    predictions = model.predict(test_ds)
    predicted_classes = np.argmax(predictions, axis=1)
    true_labels = np.concatenate([y for x, y in test_ds], axis=0)
    cm = confusion_matrix(true_labels, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - AlexNet")
    plt.tight_layout()
    plt.savefig(conf.ROOT+"/src/evaluation/mobilenetv2_evaluation.png")
    plt.show()
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_classes, target_names=class_names))
    return predicted_classes, true_labels, class_names

if __name__ == "__main__":
    evaluate_mobilenetv2()