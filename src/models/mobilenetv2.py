import utils.config as conf
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def build_mobilenetv2(input_shape=(224, 224, 3), num_classes=10):
    "Loads a pre-trained MobileNetV2 without the top classification layer."
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation="relu")(x) 
    x = Dropout(0.5)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output_layer)
    model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )
    return model

def train_mobilenetv2(train_dir=conf.DATASET_TRAIN, test_dir=conf.DATASET_TEST, save_path=conf.SAVED_MODELS_PATH, image_size=(224, 224), batch_size=64, epochs=10):
    """
    Loads dataset, trains MobileNetV2, and saves the model.
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
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    model = build_mobilenetv2(
        input_shape=(image_size[0], image_size[1], 3),
        num_classes=num_classes
    )
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds
    )
    model.save(save_path+"/mobilenetv2.h5")
    print(f"[INFO] Model saved to {save_path}")
    return model, history

if __name__ == "__main__":
    train_mobilenetv2()