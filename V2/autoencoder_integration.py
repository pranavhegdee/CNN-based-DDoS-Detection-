"""
autoencoder_integration.py
----------------------------
Drop-in module for cnn_guard1.py. Import this alongside your existing
CNN loading code and call check_anomaly(feature_vec_64) BEFORE handing
the same feature vector to the CNN. Suggested pipeline per window:

    vec64 = compute_window_features(...)                # unchanged
    is_anomalous, recon_error = anomaly_gate.check(vec64)

    if not is_anomalous:
        # Skip the CNN entirely -- this window reconstructs like normal
        # traffic, so there's nothing to classify. Saves inference cost
        # and, more importantly, means the CNN is never asked to force
        # a label onto traffic that isn't attack-shaped at all.
        report BENIGN / no action

    else:
        # Anomalous. Now ask the CNN which KNOWN type it resembles most.
        pred_class, confidence = cnn_predict(vec64)      # your existing code
        if confidence is high and pred_class != BENIGN:
            # matches a known attack signature -> existing mitigation path
            ...
        else:
            # Anomalous but the CNN doesn't confidently recognize it as
            # one of its 3 known types either -> flag as UNKNOWN_ANOMALY
            # for operator review, same label already used by the
            # volumetric bypass gate, but now for a genuinely different
            # and useful reason: "this is something new."
            ...

This two-stage design means an entirely new attack type (something not
SYN flood, not UDP flood, not benign -- e.g. an ICMP flood, a port scan,
whatever comes next) still gets FLAGGED, even though the CNN alone would
have no way to say anything other than "closest known label, and it's
probably wrong."
"""

import json
import pickle
import numpy as np

from autoencoder_arch import build_autoencoder, INPUT_DIM


class AnomalyGate:
    def __init__(self, weights_path="autoencoder.weights.h5",
                 scaler_path="ae_scaler.pkl", threshold_path="ae_threshold.json"):
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(threshold_path) as f:
            meta = json.load(f)
        self.threshold = meta["threshold"]
        self.model = build_autoencoder(input_dim=INPUT_DIM, bottleneck=8)
        self.model.load_weights(weights_path)
        print(f"[*] AnomalyGate loaded. Threshold (p99 benign reconstruction MSE) = {self.threshold:.5f}")

    def check(self, vec64):
        """vec64: the same (1, 64) feature vector compute_window_features()
        already produces for the CNN. Returns (is_anomalous: bool, error: float)."""
        vec41 = vec64[:, :INPUT_DIM]
        scaled = self.scaler.transform(vec41)
        recon = self.model.predict(scaled, verbose=0)
        error = float(np.mean(np.square(scaled - recon)))
        return error > self.threshold, error


# Example wiring (add near your existing model-loading code in cnn_guard1.py):
#
#   from autoencoder_integration import AnomalyGate
#   anomaly_gate = AnomalyGate()
#
#   ... inside your per-window detection loop, BEFORE the existing CNN call ...
#   is_anomalous, recon_error = anomaly_gate.check(feature_vector)
#   print(f"[AE] reconstruction_error={recon_error:.5f} anomalous={is_anomalous}")
#
#   if not is_anomalous:
#       # fast path: skip CNN, treat as BENIGN, continue to next window
#       continue
#   # else fall through to your existing CNN classification + gating logic unchanged
