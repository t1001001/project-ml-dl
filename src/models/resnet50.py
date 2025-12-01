import utils.config as conf
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model

def build_resnet50(input_shape=(224, 224, 3), num_classes=10):
    "Loads a pre-trained ResNet50 without the top classification layer."
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )
    return model

def train_resnet50(train_dir=conf.DATASET_TRAIN, test_dir=conf.DATASET_TEST, save_path=conf.SAVED_MODELS_PATH, image_size=(224, 224), batch_size=64, epochs=10):
    """
    Loads dataset, trains ResNet50, and saves the model.
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
    model = build_resnet50(
        input_shape=(image_size[0], image_size[1], 3),
        num_classes=num_classes
    )
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds
    )
    model.save(save_path+"resnet50.h5")
    print(f"[INFO] Model saved to {save_path}")
    return model, history

if __name__ == "__main__":
    train_resnet50()