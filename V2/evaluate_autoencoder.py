import json, pickle
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autoencoder_arch import build_autoencoder, INPUT_DIM
from model_arch import build_model
import stress_test_dataset as st
from novel_attack_dataset import gen_icmp_flood
from generate_dataset import gen_benign

st.RNG = np.random.default_rng(31415)  # fresh, unseen seed

with open("ae_threshold.json") as f:
    ae_meta = json.load(f)
THRESHOLD = ae_meta["threshold"]

with open("ae_scaler.pkl", "rb") as f:
    ae_scaler = pickle.load(f)
autoencoder = build_autoencoder(input_dim=INPUT_DIM, bottleneck=8)
autoencoder.load_weights("autoencoder.weights.h5")

with open("labels.json") as f:
    CLASS_NAMES = json.load(f)
with open("scaler.pkl", "rb") as f:
    cnn_scaler = pickle.load(f)
cnn_model = build_model(num_classes=len(CLASS_NAMES))
cnn_model.load_weights("traffic_cnn_model_premium.weights.h5")


def ae_scores(X64):
    X41 = X64[:, :INPUT_DIM]
    Xs = ae_scaler.transform(X41)
    recon = autoencoder.predict(Xs, verbose=0)
    return np.mean(np.square(Xs - recon), axis=1)


def cnn_predict(X64):
    Xs = cnn_scaler.transform(X64).reshape(-1, 8, 8, 1)
    probs = cnn_model.predict(Xs, verbose=0)
    return np.argmax(probs, axis=1), probs


# ---------------------------------------------------------------
# Part A: benign vs known-attack-type separation (sanity + ROC-AUC)
# ---------------------------------------------------------------
X_benign_fresh = gen_benign(1000)
X_syn, y_syn = st.gen_syn_flood_throttled(500), None
X_udp = st.gen_udp_flood_throttled(500)
X_known_attacks = np.vstack([X_syn, X_udp])

benign_err = ae_scores(X_benign_fresh)
attack_err = ae_scores(X_known_attacks)

y_true_binary = np.array([0] * len(benign_err) + [1] * len(attack_err))
all_err = np.concatenate([benign_err, attack_err])
auc_known = roc_auc_score(y_true_binary, all_err)

y_pred_binary = (all_err > THRESHOLD).astype(int)
prec, rec, f1, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, average="binary")
cm_known = confusion_matrix(y_true_binary, y_pred_binary)

print("=" * 65)
print("PART A: Autoencoder — benign vs KNOWN attack types (SYN/UDP, throttled)")
print("=" * 65)
print(f"ROC-AUC: {auc_known:.4f}")
print(f"At threshold={THRESHOLD:.5f}: precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")
print("Confusion matrix [rows=true(0=benign,1=attack), cols=pred]:")
print(cm_known)

# ---------------------------------------------------------------
# Part B: THE KEY TEST — novel attack type neither model has trained on
# ---------------------------------------------------------------
X_icmp = gen_icmp_flood(500)
icmp_err = ae_scores(X_icmp)
icmp_flagged_rate = float(np.mean(icmp_err > THRESHOLD))

cnn_pred_icmp, cnn_probs_icmp = cnn_predict(X_icmp)
cnn_pred_counts = {CLASS_NAMES[i]: int(np.sum(cnn_pred_icmp == i)) for i in range(len(CLASS_NAMES))}
cnn_avg_conf = float(np.mean(np.max(cnn_probs_icmp, axis=1)))

print("\n" + "=" * 65)
print("PART B: Novel, never-trained attack type — ICMP flood")
print("=" * 65)
print(f"Autoencoder: {icmp_flagged_rate*100:.2f}% of ICMP-flood windows flagged as anomalous")
print(f"             mean reconstruction error = {np.mean(icmp_err):.5f} (benign mean = {np.mean(benign_err):.5f}, threshold = {THRESHOLD:.5f})")
print(f"CNN: forced to pick one of its 3 known labels regardless of fit —")
print(f"     prediction breakdown: {cnn_pred_counts}")
print(f"     average confidence on these WRONG-by-construction predictions: {cnn_avg_conf*100:.2f}%")

# ---------------------------------------------------------------
# Figures
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Error distributions
axes[0].hist(benign_err, bins=40, alpha=0.6, label="BENIGN", color="#2ca02c", density=True)
axes[0].hist(attack_err, bins=40, alpha=0.6, label="Known attacks (SYN/UDP, throttled)", color="#d62728", density=True)
axes[0].hist(icmp_err, bins=40, alpha=0.6, label="Novel: ICMP flood (unseen)", color="#9467bd", density=True)
axes[0].axvline(THRESHOLD, color="black", linestyle="--", label=f"Threshold (p99 benign)={THRESHOLD:.4f}")
axes[0].set_xlabel("Reconstruction MSE"); axes[0].set_ylabel("Density")
axes[0].set_title("Autoencoder reconstruction error by traffic type")
axes[0].legend(fontsize=8)
axes[0].set_xlim(0, np.percentile(np.concatenate([benign_err, attack_err, icmp_err]), 99))

# ROC curve for known-attack separation
fpr, tpr, _ = roc_curve(y_true_binary, all_err)
axes[1].plot(fpr, tpr, label=f"Autoencoder (AUC={auc_known:.3f})", color="#1f77b4")
axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC — Autoencoder anomaly detection (known attacks)")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig("autoencoder_evaluation.png", dpi=150)

results = {
    "known_attack_separation": {
        "roc_auc": float(auc_known), "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "threshold": float(THRESHOLD), "confusion_matrix": cm_known.tolist(),
    },
    "novel_icmp_flood_generalization": {
        "flagged_anomalous_rate": icmp_flagged_rate,
        "mean_reconstruction_error": float(np.mean(icmp_err)),
        "benign_mean_reconstruction_error": float(np.mean(benign_err)),
        "cnn_forced_prediction_breakdown": cnn_pred_counts,
        "cnn_mean_confidence_on_unseen_type": cnn_avg_conf,
    },
}
with open("autoencoder_eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved autoencoder_evaluation.png and autoencoder_eval_results.json")
