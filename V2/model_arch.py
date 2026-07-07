"""
model_arch.py
--------------
Defines the CNN architecture as plain Python code, shared by train_model.py
and cnn_guard.py.

Why this file exists: saving a full Keras model (architecture + weights)
into .h5 bakes in serialized layer configs, including the exact
initializer class signature (e.g. GlorotUniform's constructor arguments).
Those signatures can change between Keras versions, so a model saved by
one Keras version can fail to *deserialize* on a machine with a different
Keras version, even though nothing is actually wrong with the model
itself. That's exactly what happened here: trained with Keras 3.15, and
an older installed Keras doesn't yet accept
GlorotUniform(input_axes=..., output_axes=...).

The fix: never deserialize layer configs across environments. Build the
architecture fresh from this function every time (stable, version-proof),
and only transfer the numeric weight tensors via `model.save_weights()` /
`model.load_weights()`, which don't carry any Python object configs.
"""

import tensorflow as tf


def build_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8, 8, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
