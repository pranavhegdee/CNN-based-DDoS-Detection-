"""
capture_training_data.py
-------------------------
Builds a REAL (not synthetic) training dataset by running this on the
target host's namespace while you separately generate each type of traffic
in Mininet. It uses the exact same compute_window_features() your live
detector uses, so the model trains on the same feature distribution it
will see at inference time — closing the synthetic-vs-real gap that's
causing "everything says BENIGN/UNKNOWN_ANOMALY" right now.

WORKFLOW
--------
1. Start the target server first (see commands at the bottom).
2. Run this script with the label for whatever traffic you're ABOUT to
   generate, e.g.:

     mininet> h1 python3 capture_training_data.py BENIGN 120

   (label, duration_seconds). It captures 1-second windows for that long
   and appends rows to real_dataset.csv with that label.

3. In another Mininet terminal, generate the matching attack traffic
   AT THE SAME TIME (commands at the bottom of this file).

4. Repeat for each class: BENIGN, SLOWLORIS, SYN_FLOOD, UDP_FLOOD.
   Capture at least a few minutes per class, ideally with some variation
   (different socket counts / packet rates / durations) so the model
   generalizes instead of memorizing one exact traffic shape.

5. Run train_model.py on the resulting real_dataset.csv.
"""

import sys
import csv
import os
import time
from collections import Counter
from scapy.all import sniff, IP, TCP

from feature_extraction import compute_window_features, FEATURE_NAMES

IFACE = os.environ.get("CAPTURE_IFACE", "s1-eth1")  # match your cnn_guard1.py IFACE
OUT_CSV = "real_dataset.csv"
FWD_DST_IP = os.environ.get("FWD_DST_IP", "10.0.0.1")  # match cnn_guard1.py's is_fwd check


def capture_one_window():
    pkt_count = fwd_pkts = bwd_pkts = fwd_len = bwd_len = 0
    max_fwd_len = min_fwd_len = max_bwd_len = min_bwd_len = 0
    packet_sizes, src_ips = [], []
    fwd_lengths, bwd_lengths = [], []
    timestamps, fwd_timestamps, bwd_timestamps = [], [], []
    flag_counts = Counter()

    def process(packet):
        nonlocal pkt_count, fwd_pkts, bwd_pkts, fwd_len, bwd_len
        nonlocal max_fwd_len, min_fwd_len, max_bwd_len, min_bwd_len
        if not packet.haslayer(IP):
            return
        pkt_count += 1
        p_len = len(packet)
        p_time = float(packet.time)
        packet_sizes.append(p_len)
        src_ips.append(packet[IP].src)
        timestamps.append(p_time)

        is_fwd = packet[IP].dst == FWD_DST_IP
        if is_fwd:
            fwd_pkts += 1
            fwd_len += p_len
            fwd_lengths.append(p_len)
            fwd_timestamps.append(p_time)
            max_fwd_len = max(max_fwd_len, p_len)
            min_fwd_len = p_len if min_fwd_len == 0 else min(min_fwd_len, p_len)
        else:
            bwd_pkts += 1
            bwd_len += p_len
            bwd_lengths.append(p_len)
            bwd_timestamps.append(p_time)
            max_bwd_len = max(max_bwd_len, p_len)
            min_bwd_len = p_len if min_bwd_len == 0 else min(min_bwd_len, p_len)

        if packet.haslayer(TCP):
            flags = packet[TCP].flags
            if flags & 0x02: flag_counts['SYN'] += 1
            if flags & 0x10: flag_counts['ACK'] += 1
            if flags & 0x01: flag_counts['FIN'] += 1
            if flags & 0x04: flag_counts['RST'] += 1
            if flags & 0x08: flag_counts['PSH'] += 1
            if flags & 0x20: flag_counts['URG'] += 1

    sniff(iface=IFACE, timeout=1, prn=process, store=False)

    if pkt_count == 0:
        return None

    vec = compute_window_features(
        fwd_pkts=fwd_pkts, bwd_pkts=bwd_pkts, fwd_len=fwd_len, bwd_len=bwd_len,
        max_fwd_len=max_fwd_len, min_fwd_len=min_fwd_len,
        max_bwd_len=max_bwd_len, min_bwd_len=min_bwd_len,
        packet_sizes=packet_sizes, fwd_lengths=fwd_lengths, bwd_lengths=bwd_lengths,
        timestamps=timestamps, fwd_timestamps=fwd_timestamps, bwd_timestamps=bwd_timestamps,
        flag_counts=flag_counts,
    )
    return vec.reshape(-1)  # length 64


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 capture_training_data.py <LABEL> <DURATION_SECONDS>")
        print("  e.g. python3 capture_training_data.py SLOWLORIS 120")
        sys.exit(1)

    label = sys.argv[1]
    duration = int(sys.argv[2])

    file_exists = os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            # 64 columns: 41 named + f41..f63 padding, then label
            header = list(FEATURE_NAMES) + [f"f{i}" for i in range(41, 64)] + ["label"]
            writer.writerow(header)

        start = time.time()
        n_windows = 0
        n_empty = 0
        while time.time() - start < duration:
            vec = capture_one_window()
            if vec is None:
                n_empty += 1
                continue
            writer.writerow(list(vec) + [label])
            f.flush()
            n_windows += 1

        print(f"[*] Captured {n_windows} windows for label '{label}' "
              f"({n_empty} empty/idle windows skipped) -> {OUT_CSV}")


if __name__ == "__main__":
    main()
