"""
autoencoder_arch.py
--------------------
Architecture for the unsupervised anomaly-detection companion to the CNN
classifier. Trained ONLY on BENIGN traffic; reconstruction error at
inference time is the anomaly score. Operates on the 41 REAL features
(not the zero-padded 64-length CNN input) since a dense autoencoder has
no reason to carry 23 dead zero columns -- that's purely an artifact of
needing a square image for the CNN.

Same version-proofing rationale as model_arch.py: save weights only,
rebuild architecture from this function at load time.
"""

import tensorflow as tf

INPUT_DIM = 41  # N_REAL_FEATURES from feature_extraction.py


def build_autoencoder(input_dim=INPUT_DIM, bottleneck=8):
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(32, activation="relu")(inputs)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    bottleneck_layer = tf.keras.layers.Dense(bottleneck, activation="relu", name="bottleneck")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(bottleneck_layer)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(input_dim, activation="linear")(x)

    autoencoder = tf.keras.Model(inputs, outputs, name="benign_autoencoder")
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return autoencoder
