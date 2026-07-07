"""
feature_extraction.py
----------------------
Single source of truth for turning a 1-second traffic window into the
64-slot feature vector the CNN consumes. Used by BOTH:

  - cnn_guard.py         (real packets captured live from Mininet)
  - generate_dataset.py  (synthetic packets simulated for training)

Keeping this logic in one place is the whole point: if training features
and live-inference features are computed by two different code paths that
drift apart even slightly, the model's accuracy on synthetic data means
nothing once it sees real traffic. Both callers must go through
`compute_window_features()`.

Feature layout (indices 0-40 are real signal, 41-63 reserved/zero):
  0  flow_duration            11 fwd_len_mean        27 syn_count
  1  fwd_pkts                 12 fwd_len_std          28 ack_count
  2  bwd_pkts                 13 bwd_len_mean          29 fin_count
  3  fwd_len                  14 bwd_len_std          30 rst_count
  4  bwd_len                  15 flow_iat_mean         31 psh_count
  5  max_fwd_len               16 flow_iat_std          32 urg_count
  6  min_fwd_len               17 flow_iat_max          33 fwd_pkts_per_s
  7  max_bwd_len               18 flow_iat_min          34 bwd_pkts_per_s
  8  min_bwd_len               19 fwd_iat_mean          35 down_up_ratio
  9  flow_bytes_per_s          20 fwd_iat_std           36 pkt_len_min
  10 flow_pkts_per_s           21 fwd_iat_max           37 pkt_len_max
                                22 fwd_iat_min           38 pkt_len_std
                                23 bwd_iat_mean          39 pkt_len_var
                                24 bwd_iat_std           40 avg_packet_size
                                25 bwd_iat_max
                                26 bwd_iat_min
"""

import numpy as np

N_FEATURES = 64
N_REAL_FEATURES = 41

FEATURE_NAMES = [
    "flow_duration", "fwd_pkts", "bwd_pkts", "fwd_len", "bwd_len",
    "max_fwd_len", "min_fwd_len", "max_bwd_len", "min_bwd_len",
    "flow_bytes_per_s", "flow_pkts_per_s",
    "fwd_len_mean", "fwd_len_std", "bwd_len_mean", "bwd_len_std",
    "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min",
    "fwd_iat_mean", "fwd_iat_std", "fwd_iat_max", "fwd_iat_min",
    "bwd_iat_mean", "bwd_iat_std", "bwd_iat_max", "bwd_iat_min",
    "syn_count", "ack_count", "fin_count", "rst_count", "psh_count", "urg_count",
    "fwd_pkts_per_s", "bwd_pkts_per_s", "down_up_ratio",
    "pkt_len_min", "pkt_len_max", "pkt_len_std", "pkt_len_var", "avg_packet_size",
]
assert len(FEATURE_NAMES) == N_REAL_FEATURES


def iat_stats(sorted_times):
    """Inter-arrival-time mean/std/max/min from a list of timestamps."""
    if len(sorted_times) < 2:
        return 0.0, 0.0, 0.0, 0.0
    times = np.sort(np.asarray(sorted_times, dtype=float))
    diffs = np.diff(times)
    return float(diffs.mean()), float(diffs.std()), float(diffs.max()), float(diffs.min())


def compute_window_features(
    fwd_pkts, bwd_pkts, fwd_len, bwd_len,
    max_fwd_len, min_fwd_len, max_bwd_len, min_bwd_len,
    packet_sizes, fwd_lengths, bwd_lengths,
    timestamps, fwd_timestamps, bwd_timestamps,
    flag_counts, flow_duration=1.0,
):
    """Build the 64-length feature vector for one capture window.

    All arguments are the same raw aggregates cnn_guard.py already collects
    per-window (packet counts, byte totals, per-direction length lists,
    per-direction timestamp lists, and a flag Counter with keys
    SYN/ACK/FIN/RST/PSH/URG). Returns shape (1, 64) float64 array.
    """
    pkt_count = fwd_pkts + bwd_pkts
    packet_sizes = packet_sizes if len(packet_sizes) else [0.0]

    avg_packet_size = float(np.mean(packet_sizes))
    pkt_len_std = float(np.std(packet_sizes))
    pkt_len_var = float(np.var(packet_sizes))
    pkt_len_min = float(np.min(packet_sizes))
    pkt_len_max = float(np.max(packet_sizes))

    fwd_len_mean = float(np.mean(fwd_lengths)) if len(fwd_lengths) else 0.0
    fwd_len_std = float(np.std(fwd_lengths)) if len(fwd_lengths) else 0.0
    bwd_len_mean = float(np.mean(bwd_lengths)) if len(bwd_lengths) else 0.0
    bwd_len_std = float(np.std(bwd_lengths)) if len(bwd_lengths) else 0.0

    flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = iat_stats(timestamps)
    fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = iat_stats(fwd_timestamps)
    bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = iat_stats(bwd_timestamps)

    fwd_pkts_per_s = fwd_pkts / flow_duration
    bwd_pkts_per_s = bwd_pkts / flow_duration
    down_up_ratio = (bwd_pkts / fwd_pkts) if fwd_pkts > 0 else 0.0

    vec = np.zeros(N_FEATURES, dtype=float)
    vec[0] = flow_duration
    vec[1] = fwd_pkts
    vec[2] = bwd_pkts
    vec[3] = fwd_len
    vec[4] = bwd_len
    vec[5] = max_fwd_len
    vec[6] = min_fwd_len
    vec[7] = max_bwd_len
    vec[8] = min_bwd_len
    vec[9] = (fwd_len + bwd_len) / flow_duration
    vec[10] = pkt_count / flow_duration
    vec[11] = fwd_len_mean
    vec[12] = fwd_len_std
    vec[13] = bwd_len_mean
    vec[14] = bwd_len_std
    vec[15] = flow_iat_mean
    vec[16] = flow_iat_std
    vec[17] = flow_iat_max
    vec[18] = flow_iat_min
    vec[19] = fwd_iat_mean
    vec[20] = fwd_iat_std
    vec[21] = fwd_iat_max
    vec[22] = fwd_iat_min
    vec[23] = bwd_iat_mean
    vec[24] = bwd_iat_std
    vec[25] = bwd_iat_max
    vec[26] = bwd_iat_min
    vec[27] = flag_counts.get('SYN', 0)
    vec[28] = flag_counts.get('ACK', 0)
    vec[29] = flag_counts.get('FIN', 0)
    vec[30] = flag_counts.get('RST', 0)
    vec[31] = flag_counts.get('PSH', 0)
    vec[32] = flag_counts.get('URG', 0)
    vec[33] = fwd_pkts_per_s
    vec[34] = bwd_pkts_per_s
    vec[35] = down_up_ratio
    vec[36] = pkt_len_min
    vec[37] = pkt_len_max
    vec[38] = pkt_len_std
    vec[39] = pkt_len_var
    vec[40] = avg_packet_size

    return vec.reshape(1, N_FEATURES)
