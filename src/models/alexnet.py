import utils.config as conf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_alexnet(input_shape=(224, 224, 3), num_classes=10):
    """Builds and returns the AlexNet model."""
    model = Sequential([
        Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=input_shape),
        MaxPooling2D((3, 3), strides=2),
        Conv2D(256, (5, 5), padding='same', activation='relu'),
        MaxPooling2D((3, 3), strides=2),
        Conv2D(384, (3, 3), padding='same', activation='relu'),
        Conv2D(384, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((3, 3), strides=2),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_alexnet(train_dir=conf.DATASET_TRAIN, test_dir=conf.DATASET_TEST, save_path=conf.SAVED_MODELS_PATH, image_size=(224, 224), batch_size=64, epochs=10):
    """
    Loads dataset, trains AlexNet, and saves the model.
    """
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size
    )
    num_classes = len(train_ds.class_names)
    print("Detected classes:", train_ds.class_names)
    model = build_alexnet(
        input_shape=(image_size[0], image_size[1], 3),
        num_classes=num_classes
    )
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds
    )
    model.save(save_path+"alexnet.h5")
    print(f"[INFO] Model saved to {save_path}")
    return model, history

if __name__ == "__main__":
    train_alexnet()