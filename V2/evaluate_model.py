"""
evaluate_model.py
------------------
Full report on the CNN traffic-guard model, run against your real dataset:
  1. Parameter counts (total / trainable / non-trainable, per-layer table + chart)
  2. Training curves (accuracy/loss) IF history.json exists (optional)
  3. Confusion matrix + accuracy/precision/recall/F1 (per-class and overall)
     computed on dataset_ddos.csv using the SAME feature order as training.

Usage:
    python3 evaluate_model.py
"""

import os
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support,
)

from model_arch import build_model
from feature_extraction import FEATURE_NAMES, N_FEATURES

# ---------------------------- CONFIG -------------------------------------
WEIGHTS_PATH = "weights.h5"
LABELS_PATH = "labels.json"
SCALER_PATH = "scaler.pkl"
HISTORY_PATH = "history.json"          # optional
DATASET_CSV = "dataset.csv"
OUTPUT_DIR = "eval_report"
TEST_SPLIT = 0.2                       # held-out fraction for evaluation
RANDOM_SEED = 42
# ---------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_weights_by_name(model, weights_path):
    """Robust loader that reads the Keras-3 native weights.h5 layout
    (layers/<name>/vars/<i>) directly, bypassing model.load_weights().

    Falls back to this when model.load_weights() rejects the file because
    it's missing the top-level format/version attrs Keras uses to pick a
    loader path (seen when a .weights.h5 gets re-saved/re-uploaded without
    its HDF5 attributes intact) — the actual weight arrays are unaffected,
    so we can assign them straight onto each layer by name.
    """
    import h5py
    with h5py.File(weights_path, "r") as f:
        layer_group = f["layers"]
        for layer in model.layers:
            if layer.name not in layer_group:
                continue
            var_group = layer_group[layer.name].get("vars")
            if var_group is None or len(layer.weights) == 0:
                continue
            arrays = [np.asarray(var_group[str(i)]) for i in range(len(var_group))]
            if len(arrays) != len(layer.weights):
                raise ValueError(
                    f"Layer '{layer.name}' expects {len(layer.weights)} weight "
                    f"tensors but file has {len(arrays)}"
                )
            layer.set_weights(arrays)


def load_model_and_labels():
    with open(LABELS_PATH) as f:
        class_names = json.load(f)
    model = build_model(num_classes=len(class_names))
    try:
        model.load_weights(WEIGHTS_PATH)
    except ValueError as e:
        print(f"[note] model.load_weights() failed ({e}); "
              f"loading weight tensors directly by layer name instead.")
        _load_weights_by_name(model, WEIGHTS_PATH)
    return model, class_names


# --------------------------- 1. PARAMETERS --------------------------------

def report_parameters(model):
    print("\n" + "=" * 70)
    print("MODEL PARAMETERS")
    print("=" * 70)
    model.summary()

    total_params = model.count_params()
    trainable = sum(int(tf.size(w)) for w in model.trainable_weights)
    non_trainable = sum(int(tf.size(w)) for w in model.non_trainable_weights)

    print(f"\nTotal params:         {total_params:,}")
    print(f"Trainable params:     {trainable:,}")
    print(f"Non-trainable params: {non_trainable:,}")

    rows = [(l.name, l.__class__.__name__, l.count_params())
            for l in model.layers if l.count_params() > 0]

    print(f"\n{'Layer':<24}{'Type':<22}{'Params':>12}")
    print("-" * 58)
    for name, ltype, n in rows:
        print(f"{name:<24}{ltype:<22}{n:>12,}")

    names = [r[0] for r in rows]
    counts = [r[2] for r in rows]
    plt.figure(figsize=(9, 5))
    plt.bar(names, counts, color="#4C72B0")
    plt.ylabel("Parameter count")
    plt.title("Parameters per layer")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "params_per_layer.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n[saved] {out_path}")

    return {"total": total_params, "trainable": trainable, "non_trainable": non_trainable,
            "per_layer": rows}


# --------------------------- 2. TRAINING CURVES ---------------------------

def report_training_curves():
    if not os.path.exists(HISTORY_PATH):
        print(f"\n[skip] {HISTORY_PATH} not found — no training curves. "
              f"Save model.fit(...).history to JSON during training to enable this:\n"
              f'    history = model.fit(...)\n'
              f'    json.dump(history.history, open("history.json", "w"))')
        return

    with open(HISTORY_PATH) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if "accuracy" in history:
        axes[0].plot(history["accuracy"], label="train")
    if "val_accuracy" in history:
        axes[0].plot(history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)

    if "loss" in history:
        axes[1].plot(history["loss"], label="train")
    if "val_loss" in history:
        axes[1].plot(history["val_loss"], label="val")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[saved] {out_path}")


# --------------------------- 3. LOAD + SPLIT DATASET -----------------------

def load_dataset(class_names):
    df = pd.read_csv(DATASET_CSV)

    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected feature columns: {missing}")

    # Build the full 64-wide matrix in the exact order compute_window_features()
    # produces (41 real features, then any f41..f63 padding columns present).
    ordered_cols = list(FEATURE_NAMES)
    extra_cols = [c for c in df.columns if c not in FEATURE_NAMES and c != "label"]
    extra_cols_sorted = sorted(extra_cols, key=lambda c: (len(c), c))  # f41, f42 ... in order
    ordered_cols += extra_cols_sorted

    X = df[ordered_cols].values.astype(float)
    if X.shape[1] < N_FEATURES:
        pad = np.zeros((X.shape[0], N_FEATURES - X.shape[1]))
        X = np.hstack([X, pad])
    elif X.shape[1] > N_FEATURES:
        X = X[:, :N_FEATURES]

    label_to_idx = {name: i for i, name in enumerate(class_names)}
    unknown = set(df["label"].unique()) - set(label_to_idx)
    if unknown:
        raise ValueError(f"CSV has labels not in labels.json: {unknown}")
    y = df["label"].map(label_to_idx).values

    return X, y


def train_test_split_manual(X, y, test_frac, seed):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


# --------------------------- 4. CONFUSION MATRIX + METRICS -----------------

def report_confusion_matrix(model, class_names):
    X, y = load_dataset(class_names)
    print(f"\nDataset loaded: {X.shape[0]} rows, {X.shape[1]} features, "
          f"{len(class_names)} classes")
    print("Class distribution:", {class_names[i]: int((y == i).sum()) for i in range(len(class_names))})

    _, _, X_test, y_test = train_test_split_manual(X, y, TEST_SPLIT, RANDOM_SEED)
    print(f"Evaluating on held-out test split: {len(X_test)} rows "
          f"({TEST_SPLIT*100:.0f}% of dataset, seed={RANDOM_SEED})")

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    X_scaled = scaler.transform(X_test)
    X_img = X_scaled.reshape(-1, 8, 8, 1)

    probs = model.predict(X_img, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(class_names)), zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 70)
    print("OVERALL METRICS")
    print("=" * 70)
    print(f"Accuracy:            {acc:.4f}")
    print(f"Macro    Precision:  {macro_p:.4f}   Recall: {macro_r:.4f}   F1: {macro_f1:.4f}")
    print(f"Weighted Precision:  {weighted_p:.4f}   Recall: {weighted_r:.4f}   F1: {weighted_f1:.4f}")

    print("\n" + "=" * 70)
    print("PER-CLASS METRICS")
    print("=" * 70)
    print(f"{'Class':<15}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Support':>12}")
    for i, name in enumerate(class_names):
        print(f"{name:<15}{precision[i]:>12.4f}{recall[i]:>12.4f}{f1[i]:>12.4f}{support[i]:>12}")

    print("\n" + "=" * 70)
    print("SKLEARN CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"{'':<12}" + "".join(f"{n:<12}" for n in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:<12}" + "".join(f"{v:<12}" for v in row))

    _plot_confusion(cm, class_names, normalize=False,
                     out_path=os.path.join(OUTPUT_DIR, "confusion_matrix_counts.png"))
    _plot_confusion(cm, class_names, normalize=True,
                     out_path=os.path.join(OUTPUT_DIR, "confusion_matrix_normalized.png"))

    # Also save a metrics bar chart (precision/recall/F1 per class)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(class_names))
    width = 0.25
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class Precision / Recall / F1")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "per_class_metrics.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[saved] {out_path}")

    # Save all numeric metrics to a JSON summary too
    summary = {
        "accuracy": acc,
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "weighted": {"precision": weighted_p, "recall": weighted_r, "f1": weighted_f1},
        "per_class": {
            class_names[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                              "f1": float(f1[i]), "support": int(support[i])}
            for i in range(len(class_names))
        },
        "confusion_matrix": cm.tolist(),
    }
    with open(os.path.join(OUTPUT_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {os.path.join(OUTPUT_DIR, 'metrics_summary.json')}")


def _plot_confusion(cm, class_names, normalize, out_path):
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        fmt, title = ".2f", "Confusion Matrix (row-normalized)"
    else:
        cm_display = cm
        fmt, title = "d", "Confusion Matrix (counts)"

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.1),
                                     max(5, len(class_names) * 0.9)))
    im = ax.imshow(cm_display, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)

    thresh = cm_display.max() / 2.0
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            val = cm_display[i, j]
            ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                     color="white" if val > thresh else "black", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    model, class_names = load_model_and_labels()
    report_parameters(model)
    report_training_curves()
    report_confusion_matrix(model, class_names)
    print(f"\nAll outputs saved to ./{OUTPUT_DIR}/")