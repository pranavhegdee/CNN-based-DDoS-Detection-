"""
generate_dataset.py
--------------------
Generates a synthetic, labeled dataset of 1-second traffic-window features
for 8 classes: BENIGN + 7 DDoS/attack types. Each row is produced by
simulating a plausible set of raw packets for that class (arrival times,
sizes, TCP flags, fwd/bwd split) and then running them through
feature_extraction.compute_window_features() -- the SAME function
cnn_guard.py uses on real captured traffic. That's what keeps this dataset
honest: the model trains on features computed the identical way they'll be
computed at inference time.

Being upfront about limits: this is a statistical simulation of what each
attack type *tends* to look like at the flow level (rate, packet size,
flag mix, fwd/bwd asymmetry). It is not a packet-accurate replica of real
attack tools, and a model trained only on it will likely need
recalibration against real captured traffic (see cnn_guard.py's
diagnostics) before you trust it in the field. Treat this as a strong
starting point / proof of pipeline, not a finished, field-proven detector.

Usage:
    python3 generate_dataset.py --rows-per-class 6000 --out dataset_ddos.csv
"""

import argparse
import numpy as np
import pandas as pd

from feature_extraction import compute_window_features, FEATURE_NAMES

CLASSES = [
    "BENIGN",
    "SYN_FLOOD",
    "UDP_FLOOD",
    "ICMP_FLOOD",
    "HTTP_FLOOD",
    "SLOWLORIS",
    "DNS_AMPLIFICATION",
    "PORT_SCAN",
]

# Per-class generative profile. Ranges are sampled uniformly per synthetic
# row so each class covers a spread, not a single point estimate.
PROFILES = {
    "BENIGN": dict(
        rate_lambda=(40, 150), fwd_ratio=(0.40, 0.60),
        fwd_size=(200, 700, 60, 200), bwd_size=(300, 900, 80, 250),
        flags=dict(SYN=(0.01, 0.03), ACK=(0.40, 0.60), FIN=(0.01, 0.02),
                   RST=(0.0, 0.01), PSH=(0.20, 0.40), URG=(0.0, 0.005)),
        iat_variance=1.0,
    ),
    "SYN_FLOOD": dict(
        rate_lambda=(1500, 5000), fwd_ratio=(0.90, 1.0),
        fwd_size=(56, 66, 4, 8), bwd_size=(56, 66, 4, 8),
        flags=dict(SYN=(0.85, 0.98), ACK=(0.0, 0.05), FIN=(0.0, 0.0),
                   RST=(0.0, 0.02), PSH=(0.0, 0.0), URG=(0.0, 0.0)),
        iat_variance=0.4,
    ),
    "UDP_FLOOD": dict(
        rate_lambda=(1800, 4500), fwd_ratio=(0.85, 1.0),
        fwd_size=(500, 1400, 100, 300), bwd_size=(40, 100, 10, 30),
        flags=dict(SYN=(0, 0), ACK=(0, 0), FIN=(0, 0),
                   RST=(0, 0), PSH=(0, 0), URG=(0, 0)),
        iat_variance=0.5,
    ),
    "ICMP_FLOOD": dict(
        rate_lambda=(1500, 3800), fwd_ratio=(0.90, 1.0),
        fwd_size=(300, 1472, 100, 300), bwd_size=(300, 1472, 100, 300),
        flags=dict(SYN=(0, 0), ACK=(0, 0), FIN=(0, 0),
                   RST=(0, 0), PSH=(0, 0), URG=(0, 0)),
        iat_variance=0.4,
    ),
    "HTTP_FLOOD": dict(
        rate_lambda=(400, 1600), fwd_ratio=(0.45, 0.70),
        fwd_size=(200, 500, 60, 150), bwd_size=(400, 1000, 150, 350),
        flags=dict(SYN=(0.02, 0.05), ACK=(0.50, 0.70), FIN=(0.0, 0.02),
                   RST=(0.0, 0.02), PSH=(0.30, 0.50), URG=(0.0, 0.0)),
        iat_variance=0.7,
    ),
    "SLOWLORIS": dict(
        rate_lambda=(4, 20), fwd_ratio=(0.70, 0.95),
        fwd_size=(50, 120, 15, 40), bwd_size=(40, 80, 10, 20),
        flags=dict(SYN=(0.05, 0.15), ACK=(0.30, 0.50), FIN=(0.0, 0.01),
                   RST=(0.0, 0.01), PSH=(0.10, 0.20), URG=(0.0, 0.0)),
        iat_variance=3.0,   # long, irregular gaps by design
    ),
    "DNS_AMPLIFICATION": dict(
        rate_lambda=(800, 2500), fwd_ratio=(0.40, 0.60),
        fwd_size=(50, 70, 5, 10), bwd_size=(1800, 3200, 400, 900),
        flags=dict(SYN=(0, 0), ACK=(0, 0), FIN=(0, 0),
                   RST=(0, 0), PSH=(0, 0), URG=(0, 0)),
        iat_variance=0.6,
    ),
    "PORT_SCAN": dict(
        rate_lambda=(100, 450), fwd_ratio=(0.45, 0.60),
        fwd_size=(50, 66, 5, 10), bwd_size=(50, 70, 5, 15),
        flags=dict(SYN=(0.30, 0.50), ACK=(0.05, 0.15), FIN=(0.0, 0.01),
                   RST=(0.30, 0.50), PSH=(0.0, 0.0), URG=(0.0, 0.0)),
        iat_variance=1.2,
    ),
}


def _sample_range(rng, lo_hi):
    lo, hi = lo_hi
    return rng.uniform(lo, hi) if hi > lo else lo


def _sample_sizes(rng, spec, n):
    lo_mean, hi_mean, lo_std, hi_std = spec
    mean = rng.uniform(lo_mean, hi_mean)
    std = rng.uniform(lo_std, hi_std)
    sizes = rng.normal(mean, max(std, 1e-3), size=n)
    return np.clip(sizes, 20, 9000)


def _sample_arrival_times(rng, n, iat_variance):
    """Poisson-like arrival process within a 1-second window, with a
    class-specific variance multiplier so bursty/regular attacks differ
    in their IAT statistics, not just their raw rate."""
    if n <= 1:
        return np.array([0.5] * max(n, 1))
    shape = 1.0 / max(iat_variance, 0.05)   # smaller shape -> heavier-tailed gaps
    gaps = rng.gamma(shape=shape, scale=iat_variance, size=n - 1)
    times = np.concatenate([[0.0], np.cumsum(gaps)])
    times = times / (times[-1] + 1e-9)  # normalize into [0, 1)
    return times


def simulate_window(rng, cls):
    profile = PROFILES[cls]

    total_pkts = int(rng.poisson(_sample_range(rng, profile["rate_lambda"])))
    total_pkts = max(total_pkts, 1)

    fwd_ratio = _sample_range(rng, profile["fwd_ratio"])
    fwd_pkts = max(int(total_pkts * fwd_ratio), 0)
    bwd_pkts = max(total_pkts - fwd_pkts, 0)
    if fwd_pkts + bwd_pkts == 0:
        fwd_pkts = 1

    fwd_lengths = _sample_sizes(rng, profile["fwd_size"], fwd_pkts) if fwd_pkts else np.array([])
    bwd_lengths = _sample_sizes(rng, profile["bwd_size"], bwd_pkts) if bwd_pkts else np.array([])
    packet_sizes = np.concatenate([fwd_lengths, bwd_lengths]) if (fwd_pkts + bwd_pkts) else np.array([0.0])

    fwd_timestamps = _sample_arrival_times(rng, fwd_pkts, profile["iat_variance"]) if fwd_pkts else np.array([])
    bwd_timestamps = _sample_arrival_times(rng, bwd_pkts, profile["iat_variance"]) if bwd_pkts else np.array([])
    timestamps = np.concatenate([fwd_timestamps, bwd_timestamps]) if (fwd_pkts + bwd_pkts) else np.array([0.5])

    flag_counts = {}
    for flag, rng_frac in profile["flags"].items():
        frac = _sample_range(rng, rng_frac)
        flag_counts[flag] = int(rng.poisson(max(frac * total_pkts, 0)))

    fwd_len = float(fwd_lengths.sum()) if fwd_pkts else 0.0
    bwd_len = float(bwd_lengths.sum()) if bwd_pkts else 0.0
    max_fwd_len = float(fwd_lengths.max()) if fwd_pkts else 0.0
    min_fwd_len = float(fwd_lengths.min()) if fwd_pkts else 0.0
    max_bwd_len = float(bwd_lengths.max()) if bwd_pkts else 0.0
    min_bwd_len = float(bwd_lengths.min()) if bwd_pkts else 0.0

    vec = compute_window_features(
        fwd_pkts=fwd_pkts, bwd_pkts=bwd_pkts, fwd_len=fwd_len, bwd_len=bwd_len,
        max_fwd_len=max_fwd_len, min_fwd_len=min_fwd_len,
        max_bwd_len=max_bwd_len, min_bwd_len=min_bwd_len,
        packet_sizes=list(packet_sizes), fwd_lengths=list(fwd_lengths), bwd_lengths=list(bwd_lengths),
        timestamps=list(timestamps), fwd_timestamps=list(fwd_timestamps), bwd_timestamps=list(bwd_timestamps),
        flag_counts=flag_counts,
    )
    return vec[0]


def generate(rows_per_class, seed=42):
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for cls in CLASSES:
        for _ in range(rows_per_class):
            rows.append(simulate_window(rng, cls))
            labels.append(cls)
    X = np.vstack(rows)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    # Human-readable names for the 41 real slots, to make the CSV inspectable
    for i, name in enumerate(FEATURE_NAMES):
        df.rename(columns={f"f{i}": name}, inplace=True)
    df["label"] = labels
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-per-class", type=int, default=6000)
    ap.add_argument("--out", type=str, default="dataset_ddos.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = generate(args.rows_per_class, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"[*] Wrote {len(df)} rows ({args.rows_per_class} per class x {len(CLASSES)} classes) to {args.out}")
    print(df["label"].value_counts())
