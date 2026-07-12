"""
benchmark_performance.py
--------------------------
Measures what's actually measurable from a sandbox without a live
Mininet/Ryu/OVS environment: model efficiency, inference latency, and
feature-extraction throughput. Does NOT measure real network mitigation
latency (VACL install time, actual packet drop time) -- that requires
your live testbed; see mininet_timing_benchmark.py for that half.
"""

import json
import pickle
import time
import numpy as np
from collections import Counter

from model_arch import build_model
from autoencoder_arch import build_autoencoder, INPUT_DIM
from feature_extraction import compute_window_features

with open("labels.json") as f:
    CLASS_NAMES = json.load(f)
with open("scaler.pkl", "rb") as f:
    cnn_scaler = pickle.load(f)
cnn_model = build_model(num_classes=len(CLASS_NAMES))
cnn_model.load_weights("traffic_cnn_model_premium.weights.h5")

with open("ae_scaler.pkl", "rb") as f:
    ae_scaler = pickle.load(f)
with open("ae_threshold.json") as f:
    ae_threshold = json.load(f)["threshold"]
autoencoder = build_autoencoder(input_dim=INPUT_DIM, bottleneck=8)
autoencoder.load_weights("autoencoder.weights.h5")


def count_params(model):
    return int(sum(np.prod(w.shape) for w in model.get_weights()))


def file_size_kb(path):
    import os
    return os.path.getsize(path) / 1024.0


def bench(fn, n=200, warmup=20):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000  # ms
    return {
        "mean_ms": float(times.mean()), "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)), "p99_ms": float(np.percentile(times, 99)),
        "max_ms": float(times.max()),
    }


# ---------------------------------------------------------------
# 1. Model size / efficiency
# ---------------------------------------------------------------
print("=" * 65)
print("1. MODEL SIZE / EFFICIENCY")
print("=" * 65)
cnn_params = count_params(cnn_model)
ae_params = count_params(autoencoder)
cnn_size = file_size_kb("traffic_cnn_model_premium.weights.h5")
ae_size = file_size_kb("autoencoder.weights.h5")
print(f"CNN:         {cnn_params:,} parameters, {cnn_size:.1f} KB on disk")
print(f"Autoencoder: {ae_params:,} parameters, {ae_size:.1f} KB on disk")
print(f"Combined footprint: {cnn_size + ae_size:.1f} KB — trivially small for edge/controller deployment")

# ---------------------------------------------------------------
# 2. Single-window inference latency (the number that actually matters:
#    can this keep up with a new window arriving every 1 second?)
# ---------------------------------------------------------------
print("\n" + "=" * 65)
print("2. INFERENCE LATENCY (single window, batch=1)")
print("=" * 65)

sample_vec64 = np.random.rand(1, 64).astype(np.float64)

def cnn_infer():
    scaled = cnn_scaler.transform(sample_vec64).reshape(-1, 8, 8, 1)
    cnn_model.predict(scaled, verbose=0)

def ae_infer():
    scaled = ae_scaler.transform(sample_vec64[:, :INPUT_DIM])
    autoencoder.predict(scaled, verbose=0)

def combined_infer():
    cnn_infer(); ae_infer()

cnn_lat = bench(cnn_infer)
ae_lat = bench(ae_infer)
combined_lat = bench(combined_infer)

print(f"CNN only:         mean={cnn_lat['mean_ms']:.2f}ms  p50={cnn_lat['p50_ms']:.2f}ms  "
      f"p95={cnn_lat['p95_ms']:.2f}ms  p99={cnn_lat['p99_ms']:.2f}ms")
print(f"Autoencoder only: mean={ae_lat['mean_ms']:.2f}ms  p50={ae_lat['p50_ms']:.2f}ms  "
      f"p95={ae_lat['p95_ms']:.2f}ms  p99={ae_lat['p99_ms']:.2f}ms")
print(f"Combined pipeline: mean={combined_lat['mean_ms']:.2f}ms  p50={combined_lat['p50_ms']:.2f}ms  "
      f"p95={combined_lat['p95_ms']:.2f}ms  p99={combined_lat['p99_ms']:.2f}ms")
print(f"\nWindow budget available: 1000ms (one window per second). "
      f"Combined inference uses {combined_lat['mean_ms']/1000*100:.3f}% of that budget on average "
      f"-> inference is NOT the bottleneck; the 1-second sniff window itself dominates latency.")

# ---------------------------------------------------------------
# 3. Feature-extraction throughput vs. window size (packet count)
# ---------------------------------------------------------------
print("\n" + "=" * 65)
print("3. FEATURE-EXTRACTION THROUGHPUT (compute_window_features)")
print("=" * 65)

def make_window_args(n_pkts):
    fwd = n_pkts // 2
    bwd = n_pkts - fwd
    fwd_lengths = list(np.random.randint(40, 1400, size=fwd))
    bwd_lengths = list(np.random.randint(40, 1400, size=bwd))
    packet_sizes = fwd_lengths + bwd_lengths
    timestamps = sorted(np.random.uniform(0, 1, size=n_pkts).tolist())
    fwd_ts = timestamps[:fwd]
    bwd_ts = timestamps[fwd:]
    flags = Counter(SYN=n_pkts // 3, ACK=n_pkts // 3, RST=n_pkts // 10)
    return dict(
        fwd_pkts=fwd, bwd_pkts=bwd, fwd_len=int(sum(fwd_lengths)), bwd_len=int(sum(bwd_lengths)),
        max_fwd_len=int(max(fwd_lengths)) if fwd_lengths else 0,
        min_fwd_len=int(min(fwd_lengths)) if fwd_lengths else 0,
        max_bwd_len=int(max(bwd_lengths)) if bwd_lengths else 0,
        min_bwd_len=int(min(bwd_lengths)) if bwd_lengths else 0,
        packet_sizes=packet_sizes, fwd_lengths=fwd_lengths, bwd_lengths=bwd_lengths,
        timestamps=timestamps, fwd_timestamps=fwd_ts, bwd_timestamps=bwd_ts,
        flag_counts=flags,
    )

throughput_results = {}
for n_pkts in [10, 100, 1000, 5000, 10000]:
    args = make_window_args(n_pkts)
    def call():
        compute_window_features(**args)
    lat = bench(call, n=100, warmup=10)
    pkts_per_sec_capacity = n_pkts / (lat["mean_ms"] / 1000.0)
    throughput_results[n_pkts] = {**lat, "effective_pkts_per_sec_capacity": pkts_per_sec_capacity}
    print(f"  window={n_pkts:>6} pkts:  mean={lat['mean_ms']:.3f}ms  "
          f"-> feature extraction alone could process ~{pkts_per_sec_capacity:,.0f} pkts/sec if this were the bottleneck")

# ---------------------------------------------------------------
# 4. Detection-latency budget breakdown (architectural, not measured
#    live -- these are the fixed gate timings by design)
# ---------------------------------------------------------------
print("\n" + "=" * 65)
print("4. DETECTION-LATENCY BUDGET BY TRIGGER PATH (architectural)")
print("=" * 65)
print("Volumetric/statistical bypass (pkt_count>1000 or z>4.0): 1 window  = ~1.0s + inference (~"
      f"{combined_lat['mean_ms']:.1f}ms, negligible)")
print("Persistence-confirmed model call (3 consecutive matching windows):   3 windows = ~3.0s + inference")
print("RSA token-exhaustion (failure-window based, not per-window sniff):   up to "
      "FAILURE_WINDOW_SECONDS=10s to accumulate threshold, but typically much faster under real flood rates")

# ---------------------------------------------------------------
# 5. Security metrics pulled from existing evaluation artifacts
# ---------------------------------------------------------------
print("\n" + "=" * 65)
print("5. SECURITY METRICS (from prior evaluation runs)")
print("=" * 65)
try:
    with open("eval_results_stress_v2.json") as f:
        stress = json.load(f)
    benign = stress["per_class"]["BENIGN"]
    fpr = 1 - benign["recall"]
    print(f"False Positive Rate (legitimate/BENIGN traffic misclassified as attack), "
          f"hardened model, adversarial stress test: {fpr*100:.2f}%")
    for cls in ["SYN_FLOOD", "UDP_FLOOD"]:
        fnr = 1 - stress["per_class"][cls]["recall"]
        print(f"False Negative Rate ({cls} missed): {fnr*100:.2f}%")
except FileNotFoundError:
    print("(run evaluate_stress_fresh.py first to populate this section)")

results = {
    "model_efficiency": {"cnn_params": cnn_params, "ae_params": ae_params,
                          "cnn_size_kb": cnn_size, "ae_size_kb": ae_size},
    "inference_latency_ms": {"cnn": cnn_lat, "autoencoder": ae_lat, "combined": combined_lat},
    "feature_extraction_throughput": throughput_results,
}
with open("performance_benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved performance_benchmark_results.json")
