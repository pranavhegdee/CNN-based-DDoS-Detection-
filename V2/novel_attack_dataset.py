"""
novel_attack_dataset.py
------------------------
A synthetic ICMP (ping) flood -- a type NEITHER the CNN NOR the
autoencoder has ever been trained on. Used purely to demonstrate that
the unsupervised autoencoder generalizes to unseen attack types (its
whole reason for existing), whereas the supervised CNN, by construction,
can only ever output one of its 3 known labels and has no way to say
"this is something else."
"""

import numpy as np
from collections import Counter
from generate_dataset import _window, RNG as BASE_RNG

RNG = np.random.default_rng(555)


def gen_icmp_flood(n_samples):
    X = []
    for _ in range(n_samples):
        # High-rate ICMP echo requests: no TCP flags at all (like UDP), but
        # near-uniform small packet size (unlike UDP's bare-vs-padded split)
        # and a much higher forward-only bias, since most victims don't
        # bother replying to every ping under load.
        fwd_pkts = int(RNG.integers(400, 5000))
        bwd_ratio = RNG.uniform(0.0, 0.3)  # some echo replies, but throttled under load
        bwd_pkts = int(fwd_pkts * bwd_ratio)
        fwd_len_range = (60, 84)   # standard ping payload size range
        bwd_len_range = (60, 84)
        flags = Counter()  # ICMP carries no TCP flags
        X.append(_window(fwd_pkts, bwd_pkts, fwd_len_range, bwd_len_range, flags, bursty=True))
    return np.array(X)


if __name__ == "__main__":
    X = gen_icmp_flood(500)
    np.save("X_icmp_novel.npy", X)
    print(f"[*] Novel ICMP flood set built: {X.shape}")
