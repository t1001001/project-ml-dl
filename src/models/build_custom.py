import utils.config as conf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def _build_custom(input_shape=(224, 224, 3), num_classes=10):
    """Builds and returns the custom model."""
    model = Sequential([
        Conv2D(16, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def _train_custom(train_dir=conf.DATASET_TRAIN, test_dir=conf.DATASET_TEST, save_path=conf.SAVED_MODELS_PATH, image_size=(227, 227), batch_size=64, epochs=10):
    """
    Loads dataset, trains custom, and saves the model.
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
    train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
    val_ds = val_ds.map(lambda x, y: (x / 255.0, y))
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    model = _build_custom(
        input_shape=(image_size[0], image_size[1], 3),
        num_classes=num_classes
    )
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds
    )
    model.save(save_path+"/custom.h5")
    print(f"[INFO] Model saved to {save_path}")
    return model, history

if __name__ == "__main__":
    _train_custom()