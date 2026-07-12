"""
train_autoencoder.py
---------------------
Trains the anomaly-detection autoencoder on BENIGN traffic ONLY (both the
'clean' generator and the harder overlapping/heavy-legit cases from
stress_test_dataset, so it learns a realistically wide notion of normal,
not just the easy textbook case). The threshold is calibrated from the
benign reconstruction-error distribution itself (mean + k*std, and the
99th percentile as a cross-check) rather than picked by eye.
"""

import json
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

from autoencoder_arch import build_autoencoder, INPUT_DIM
from generate_dataset import gen_benign
from stress_test_dataset import gen_benign_overlap

RNG = np.random.default_rng(77)

# ---- Build a wide benign-only training set ----
X_clean = gen_benign(6000)[:, :INPUT_DIM]        # drop the zero-padded columns
X_overlap = gen_benign_overlap(2000)[:, :INPUT_DIM]
X_benign = np.vstack([X_clean, X_overlap])
RNG.shuffle(X_benign)

X_train, X_val = train_test_split(X_benign, test_size=0.15, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

autoencoder = build_autoencoder(input_dim=INPUT_DIM, bottleneck=8)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
]

autoencoder.fit(
    X_train_scaled, X_train_scaled,
    validation_data=(X_val_scaled, X_val_scaled),
    epochs=100, batch_size=64, callbacks=callbacks, verbose=2,
)

# ---- Calibrate anomaly threshold from benign reconstruction error ----
recon = autoencoder.predict(X_val_scaled, verbose=0)
errors = np.mean(np.square(X_val_scaled - recon), axis=1)

mean_err, std_err = float(np.mean(errors)), float(np.std(errors))
p95, p99 = float(np.percentile(errors, 95)), float(np.percentile(errors, 99))
threshold_3sigma = mean_err + 3 * std_err

print(f"[*] Benign reconstruction error on validation set:")
print(f"    mean={mean_err:.5f}  std={std_err:.5f}")
print(f"    p95={p95:.5f}  p99={p99:.5f}  mean+3sigma={threshold_3sigma:.5f}")

# Use the p99 as the deployed threshold: allows ~1% benign false-positive
# rate by construction, which is a defensible, explicit operating point
# (rather than an arbitrarily-eyeballed number).
threshold = p99

autoencoder.save_weights("autoencoder.weights.h5")
with open("ae_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("ae_threshold.json", "w") as f:
    json.dump({
        "threshold": threshold,
        "mean_benign_error": mean_err,
        "std_benign_error": std_err,
        "p95": p95, "p99": p99,
        "note": "Anomaly if reconstruction MSE > threshold (p99 of benign validation error).",
    }, f, indent=2)

print(f"\n[*] Saved autoencoder.weights.h5, ae_scaler.pkl, ae_threshold.json")
print(f"[*] Deployed anomaly threshold (p99): {threshold:.5f}")
