import time
import pickle
import os
import json
import requests
import numpy as np
import tensorflow as tf
from scapy.all import sniff, IP, TCP, UDP
from collections import Counter, deque

from feature_extraction import compute_window_features, N_FEATURES
from model_arch import build_model
from autoencoder_integration import AnomalyGate
from rsa_mitigation_queue import QueueReader

with open("labels.json", "r") as f:
    CLASS_NAMES = json.load(f)   # e.g. ["BENIGN","DNS_AMPLIFICATION",...] index-ordered

print("[*] Initializing deep learning architectures...")
# Rebuild the architecture from plain Python code (model_arch.py) and load
# only the numeric weight tensors. This sidesteps a real failure mode where
# a full model.save()/.h5 load breaks across Keras versions because the
# saved file's layer configs (e.g. an initializer's exact constructor
# signature) don't match the Keras version doing the loading. Weights-only
# has no such config to deserialize.
model = build_model(num_classes=len(CLASS_NAMES))
model.load_weights("traffic_cnn_model_premium.weights.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Unsupervised companion model: trained ONLY on benign traffic, so it can
# flag anomalies that don't match ANY of the CNN's 3 known labels — the
# CNN's softmax always sums to 1 across BENIGN/SYN_FLOOD/UDP_FLOOD, so on
# an attack type it's never seen (an ICMP flood, a port scan, whatever
# comes next) it doesn't fail loudly, it fails silently and confidently
# wrong, forcing a "closest known label" guess. The autoencoder has no
# such constraint: it just measures how well this window reconstructs
# like normal traffic, so a genuinely novel attack shows up as high
# reconstruction error regardless of what the CNN thinks it looks like.
anomaly_gate = AnomalyGate(
    weights_path="autoencoder.weights.h5",
    scaler_path="ae_scaler.pkl",
    threshold_path="ae_threshold.json",
)

# gateway.py runs inside a Mininet HOST's network namespace and cannot
# reach Ryu's REST API directly (see rsa_mitigation_queue.py for why).
# This process CAN reach Ryu already, so it's the one that actually
# performs the hardware block for app-layer (RSA token exhaustion)
# escalations too, not just network-layer ones.
rsa_queue = QueueReader()

def process_rsa_mitigation_queue():
    """Poll for any IPs gateway.py has flagged for RSA token-exhaustion
    abuse and push the same hardware VACL drop rule used for network-layer
    attacks. No spoof-check needed here: unlike a packet flood, the source
    completed real TCP connections and made real HTTP requests to get
    flagged in the first place, so there's no spoofing ambiguity to
    resolve -- gateway.py already confirmed it's a real, reachable,
    abusive client."""
    for entry in rsa_queue.poll_new_entries():
        ip = entry.get("ip")
        if not ip:
            continue
        if ip in PROTECTED_IPS:
            print(f"[🛡️ RSA QUEUE] {ip} is on the protected allowlist — refusing to block "
                  f"despite {entry.get('count')} bad-signature attempts.")
            continue
        if ip in blocked_ips:
            continue  # already handled (e.g. also caught by a network-layer trigger)

        print(f"\n[🚨 ATTACK DETECTED] Type: RSA_TOKEN_EXHAUSTION | Trigger: application-layer | "
              f"Source: {ip} | {entry.get('reason', '')}")
        print(f"[📡 SIGNAL] Instructing Ryu to drop attacker: {ip}")
        try:
            requests.post("http://127.0.0.1:8080/mitigate", json={"ip": ip})
            blocked_ips.add(ip)
            report_telemetry("ATTACK", 1.0, attacker_ip=ip, new_blocked_ip=ip,
                              attack_type="RSA_TOKEN_EXHAUSTION",
                              class_probs={}, block_reason=entry.get("reason"))
        except Exception as e:
            print(f"[-] Ryu communication fault (RSA-exhaustion mitigation): {e}")

BENIGN_LABEL = "BENIGN"
ATTACK_CONFIDENCE_THRESHOLD = 0.5   # softmax prob required to trust a non-benign call

# A single 1-second window with only a few packets (e.g. a couple of pings)
# is statistically indistinguishable from a genuine low-and-slow attack
# window — both are "a few small packets." No amount of retraining fixes
# that in principle, because the information just isn't in one window.
# Two gates handle this instead of pretending the model can always be sure:
MIN_PKTS_FOR_ATTACK_CALL = 5        # rules out a literal 1-2 packet ping; persistence (below)
                                     # does the real work so this doesn't need to be aggressive
CONSECUTIVE_WINDOWS_REQUIRED = 3    # same class must repeat this many windows in a row
recent_attack_calls = deque(maxlen=CONSECUTIVE_WINDOWS_REQUIRED)

# ---- Anti-reflection / anti-spoofing protection ------------------------
# An attacker running a SYN/UDP/ICMP flood can write ANY source IP into
# their packets, including your own gateway, controller, or a real server.
# If the mitigation blindly hardware-blocks whatever IP shows up most, a
# spoofed flood can trick it into blocking your own infrastructure.
#
# Two layers of defense:
#  1. A hard allowlist that is NEVER blocked, no matter what the model or
#     any heuristic says. Put every IP you cannot afford to lock out here.
#  2. For TCP-based attack types, verify the candidate IP actually behaves
#     like a spoofed source before blocking it: a real TCP client that
#     sends a SYN will, if the server replies, eventually complete the
#     handshake with a matching ACK (or at least send further ACKs on an
#     established connection). A spoofed source can flood SYNs all day but
#     can never see the server's SYN-ACK (it isn't its real address), so it
#     can never send a completing ACK back. High SYN count + ~0 ACKs from
#     the same source is the classic signature of a spoofed reflector.
PROTECTED_IPS = set(
    ip.strip() for ip in os.environ.get("PROTECTED_IPS", "10.0.0.1").split(",") if ip.strip()
)
# Attack types where "does this source complete a TCP handshake?" is a
# meaningful question. UDP/ICMP/DNS-amplification floods have no handshake
# to check, so spoof verification doesn't apply — protection there relies
# on the allowlist + persistence gating only.
MIN_SYN_FOR_SPOOF_CHECK = 20        # need at least this many SYNs from an IP to judge it
SPOOF_ACK_RATIO_THRESHOLD = 0.05    # ACK/SYN below this looks like a one-way spoofed flood

# Cumulative (not reset per-window) so the ACK-completion signal has time
# to accumulate across the multi-second persistence window above. For a
# long-running production deployment you'd want to age these out
# periodically; fine to grow for the length of a demo/Mininet run.
syn_by_src = Counter()
ack_by_src = Counter()


def should_block_ip(ip, reported_type):
    """Decide whether it's actually safe to push a hardware DROP rule for
    this IP. Returns (allowed: bool, reason: str) so the caller can log
    exactly why a block was made or refused.

    Verification is keyed off OBSERVED SYN activity for this IP, not off
    the attack-type label. Gating on the label alone would let a volumetric
    override (e.g. the CNN says BENIGN but pkt_count > 1000, so it gets
    relabeled UNKNOWN_ANOMALY) skip verification entirely even when the
    traffic is genuinely TCP-based — the label doesn't tell you that, the
    actual SYN counter does.
    """
    if ip in PROTECTED_IPS:
        return False, f"{ip} is on the protected allowlist — refusing to block"

    syn = syn_by_src.get(ip, 0)
    if syn == 0:
        # No TCP SYN activity at all from this IP (pure UDP/ICMP/DNS
        # amplification flood) — there's no handshake to verify against.
        return True, f"no TCP SYN activity observed from {ip} — allowlist check only"

    ack = ack_by_src.get(ip, 0)
    if syn < MIN_SYN_FOR_SPOOF_CHECK:
        return False, f"only {syn} SYNs seen from {ip} so far — not enough to verify yet"
    ratio = ack / syn
    if ratio < SPOOF_ACK_RATIO_THRESHOLD:
        return True, f"verified spoofed reflector — {syn} SYNs, {ack} ACKs (ratio {ratio:.3f})"
    else:
        return False, (f"{ip} completed {ack}/{syn} handshakes (ratio {ratio:.3f}) — "
                        f"looks like a real client, not auto-blocking")
# -------------------------------------------------------------------------

# ---- RSA lockdown auto-expiry -------------------------------------------
# Previously, once /tmp/lockdown.txt was created it stayed forever until
# someone deleted it by hand. Now the file stores the timestamp of the
# LAST confirmed attack, and lifts itself automatically once this many
# seconds have passed with no further confirmed attacks — so a one-off
# incident doesn't leave the gateway locked down indefinitely. Every new
# confirmed attack refreshes the timer (sliding window from the most
# recent attack, not the first one).
LOCKDOWN_FILE = "/tmp/lockdown.txt"
LOCKDOWN_TIMEOUT_SECONDS = 60

def arm_lockdown():
    """(Re)arm the lockdown, stamping the current time. Called on every
    confirmed attack, so the 60s countdown restarts each time."""
    try:
        with open(LOCKDOWN_FILE, "w") as f:
            f.write(str(time.time()))
    except Exception as e:
        print(f"[-] Failed to write lockdown flag: {e}")

def is_lockdown_active():
    """True if lockdown is currently armed. Auto-lifts (deletes the flag
    file) once LOCKDOWN_TIMEOUT_SECONDS have passed since the last
    confirmed attack."""
    if not os.path.exists(LOCKDOWN_FILE):
        return False
    try:
        with open(LOCKDOWN_FILE) as f:
            armed_at = float(f.read().strip())
    except Exception:
        # Corrupt or legacy ("ARMED" string) file — treat as freshly armed
        # rather than crashing either way.
        arm_lockdown()
        return True
    age = time.time() - armed_at
    if age > LOCKDOWN_TIMEOUT_SECONDS:
        try:
            os.remove(LOCKDOWN_FILE)
        except Exception:
            pass
        print(f"[🔓 LOCKDOWN EXPIRED] No confirmed attacks for {age:.0f}s — "
              f"RSA lockdown lifted, gateway reopened.")
        return False
    return True
# -------------------------------------------------------------------------

# ---- Diagnostics: verify our assumptions about the model's input shape ----
print("[*] --- Feature pipeline diagnostics ---")
print(f"[*] scaler.n_features_in_ = {getattr(scaler, 'n_features_in_', 'unknown')}")
feature_names = getattr(scaler, "feature_names_in_", None)
if feature_names is not None:
    print("[*] scaler was fit with named columns — this is the REAL feature order:")
    for i, name in enumerate(feature_names):
        print(f"      [{i}] {name}")
try:
    print(f"[*] model.input_shape = {model.input_shape}")
    print(f"[*] model output classes = {model.output_shape[-1]}, labels.json has {len(CLASS_NAMES)}: {CLASS_NAMES}")
    if model.output_shape[-1] != len(CLASS_NAMES):
        print("[!] WARNING: model output size and labels.json length don't match — "
              "predictions will be mislabeled until this is fixed.")
except Exception as e:
    print(f"[-] Could not read model.input_shape/output_shape: {e}")
print("[*] --------------------------------------")
# ----------------------------------------------------------------------------

# ---- Dashboard reporting -----------------------------------------------
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://127.0.0.1:5050")

def report_telemetry(status, prediction, attacker_ip=None, new_blocked_ip=None,
                      stat_zscore=0.0, attack_type=None, class_probs=None, block_reason=None):
    if not DASHBOARD_URL:
        return
    try:
        requests.post(f"{DASHBOARD_URL}/api/telemetry", json={
            "pkt_count": pkt_count,
            "fwd_pkts": fwd_pkts,
            "bwd_pkts": bwd_pkts,
            "fwd_len": fwd_len,
            "bwd_len": bwd_len,
            "prediction": float(prediction),
            "status": status,
            "attack_type": attack_type,
            "class_probs": class_probs,
            "lockdown": is_lockdown_active(),
            "attacker_ip": attacker_ip,
            "new_blocked_ip": new_blocked_ip,
            "block_reason": block_reason,
            "blocked_ips": list(blocked_ips),
            "stat_zscore": float(stat_zscore),
        }, timeout=0.5)
    except Exception:
        pass  # dashboard offline — detection/mitigation must keep running regardless
# -------------------------------------------------------------------------

# --- Latency instrumentation: three legs, measured separately since they
# have different bottlenecks. capture_ms uses wall-clock (time.time() vs
# packet.time) since perf_counter() has no fixed epoch and can't be
# compared against a packet's own capture timestamp. ---
latency_samples = {
    "capture_ms": deque(maxlen=1000),
    "feature_ms": deque(maxlen=1000),
    "decision_ms": deque(maxlen=1000),
    "end_to_end_ms": deque(maxlen=1000),
}

def _pct(dq, p):
    if not dq:
        return float("nan")
    s = sorted(dq)
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return s[idx]

def print_latency_summary():
    print("\n[LATENCY SUMMARY]")
    for name, dq in latency_samples.items():
        if dq:
            print(f"  {name:16s} n={len(dq):4d}  mean={sum(dq)/len(dq):7.3f}ms  "
                  f"p50={_pct(dq,50):7.3f}ms  p95={_pct(dq,95):7.3f}ms  p99={_pct(dq,99):7.3f}ms")

# Global metric counters
pkt_count = fwd_pkts = bwd_pkts = fwd_len = bwd_len = max_fwd_len = min_fwd_len = max_bwd_len = min_bwd_len = 0
packet_sizes = []
src_ips = []
blocked_ips = set()

fwd_lengths = []
bwd_lengths = []
timestamps = []
fwd_timestamps = []
bwd_timestamps = []
flag_counts = Counter()

# --- Rolling statistical baseline (independent of the CNN) ---
PKT_RATE_HISTORY = deque(maxlen=30)

def pkt_rate_zscore(current_pkt_count):
    if len(PKT_RATE_HISTORY) < 10:
        return 0.0
    arr = np.array(PKT_RATE_HISTORY)
    mean, std = arr.mean(), arr.std()
    if std < 1e-6:
        return 0.0
    return (current_pkt_count - mean) / std

def process_packet(packet):
    global pkt_count, fwd_pkts, bwd_pkts, fwd_len, bwd_len, max_fwd_len, min_fwd_len, max_bwd_len, min_bwd_len
    global packet_sizes, src_ips, fwd_lengths, bwd_lengths, timestamps, fwd_timestamps, bwd_timestamps, flag_counts

    # Leg 1: capture latency -- time between the packet actually hitting the
    # wire (Scapy's own capture timestamp) and this callback running. Both
    # sides are wall-clock (epoch-based), unlike perf_counter().
    latency_samples["capture_ms"].append((time.time() - float(packet.time)) * 1000)

    if packet.haslayer(IP):
        pkt_count += 1
        p_len = len(packet)
        p_time = float(packet.time)
        packet_sizes.append(p_len)
        src_ips.append(packet[IP].src)
        timestamps.append(p_time)

        is_fwd = packet[IP].dst == "10.0.0.1"
        if is_fwd:
            fwd_pkts += 1
            fwd_len += p_len
            fwd_lengths.append(p_len)
            fwd_timestamps.append(p_time)
            if p_len > max_fwd_len: max_fwd_len = p_len
            if min_fwd_len == 0 or p_len < min_fwd_len: min_fwd_len = p_len
        else:
            bwd_pkts += 1
            bwd_len += p_len
            bwd_lengths.append(p_len)
            bwd_timestamps.append(p_time)
            if p_len > max_bwd_len: max_bwd_len = p_len
            if min_bwd_len == 0 or p_len < min_bwd_len: min_bwd_len = p_len

        if packet.haslayer(TCP):
            flags = packet[TCP].flags
            if flags & 0x02: flag_counts['SYN'] += 1
            if flags & 0x10: flag_counts['ACK'] += 1
            if flags & 0x01: flag_counts['FIN'] += 1
            if flags & 0x04: flag_counts['RST'] += 1
            if flags & 0x08: flag_counts['PSH'] += 1
            if flags & 0x20: flag_counts['URG'] += 1

            # Per-source handshake tracking for spoof verification (see
            # should_block_ip above). A pure ACK (no SYN bit) from a source
            # is that source completing/continuing a real TCP conversation
            # — something a spoofed source can never do, since it never
            # sees the server's SYN-ACK reply.
            src = packet[IP].src
            if flags & 0x02:
                syn_by_src[src] += 1
            if (flags & 0x10) and not (flags & 0x02):
                ack_by_src[src] += 1

def main_loop():
    global pkt_count, fwd_pkts, bwd_pkts, fwd_len, bwd_len, max_fwd_len, min_fwd_len, max_bwd_len, min_bwd_len
    global packet_sizes, src_ips, fwd_lengths, bwd_lengths, timestamps, fwd_timestamps, bwd_timestamps, flag_counts

    # Cheap, runs every window regardless of network-layer traffic: pick up
    # any application-layer (RSA token exhaustion) escalations gateway.py
    # has queued since the last iteration.
    process_rsa_mitigation_queue()

    # Reset tracking arrays for the new 1-second window
    pkt_count = fwd_pkts = bwd_pkts = fwd_len = bwd_len = max_fwd_len = min_fwd_len = max_bwd_len = min_bwd_len = 0
    packet_sizes = []
    src_ips = []
    fwd_lengths, bwd_lengths = [], []
    timestamps, fwd_timestamps, bwd_timestamps = [], [], []
    flag_counts = Counter()

    sniff(iface="s1-eth1", timeout=1, prn=process_packet, store=False)
    t_window_close = time.perf_counter()

    z = pkt_rate_zscore(pkt_count)
    PKT_RATE_HISTORY.append(pkt_count)

    if pkt_count == 0:
        report_telemetry("LOCKDOWN" if is_lockdown_active() else "SAFE", 0.0, stat_zscore=z)
        return

    # Leg 2: feature extraction latency
    t0 = time.perf_counter()
    # Same feature function used by generate_dataset.py during training —
    # this is what keeps live inference and training features aligned.
    raw_features_64 = compute_window_features(
        fwd_pkts=fwd_pkts, bwd_pkts=bwd_pkts, fwd_len=fwd_len, bwd_len=bwd_len,
        max_fwd_len=max_fwd_len, min_fwd_len=min_fwd_len,
        max_bwd_len=max_bwd_len, min_bwd_len=min_bwd_len,
        packet_sizes=packet_sizes, fwd_lengths=fwd_lengths, bwd_lengths=bwd_lengths,
        timestamps=timestamps, fwd_timestamps=fwd_timestamps, bwd_timestamps=bwd_timestamps,
        flag_counts=flag_counts,
    )
    latency_samples["feature_ms"].append((time.perf_counter() - t0) * 1000)

    # Leg 3: decision latency (anomaly gate + CNN inference)
    t0 = time.perf_counter()
    # Stage 1: does this window even look abnormal at all, independent of
    # what the CNN's 3 fixed labels think? Cheap (41-dim dense model), and
    # tells us whether the CNN's answer is even trustworthy to ask for.
    is_anomalous, recon_error = anomaly_gate.check(raw_features_64)

    scaled_features = scaler.transform(raw_features_64)
    traffic_image = scaled_features.reshape(-1, 8, 8, 1)

    probs = model.predict(traffic_image, verbose=0)[0]   # softmax over CLASS_NAMES
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    pred_conf = float(probs[pred_idx])
    class_probs = {name: float(p) for name, p in zip(CLASS_NAMES, probs)}
    latency_samples["decision_ms"].append((time.perf_counter() - t0) * 1000)
    latency_samples["end_to_end_ms"].append((time.perf_counter() - t_window_close) * 1000)

    if len(latency_samples["end_to_end_ms"]) % 20 == 0:
        print_latency_summary()

    print(f"[📊 SCORE] {pred_class} ({pred_conf*100:5.2f}%)  |  stat_z={z:+.2f}  |  {pkt_count} pkt/s  "
          f"(fwd={fwd_pkts} bwd={bwd_pkts} syn={flag_counts.get('SYN',0)})  |  "
          f"AE_error={recon_error:.4f}{' [ANOMALOUS]' if is_anomalous else ''}")

    # Gate 1 — sample size: a handful of packets (e.g. a couple of pings)
    # simply doesn't carry enough information for any class call to be
    # trustworthy, regardless of what the softmax says. Below this size we
    # never let the model alone trigger an attack.
    has_enough_samples = pkt_count >= MIN_PKTS_FOR_ATTACK_CALL
    model_says_attack = (pred_class != BENIGN_LABEL and pred_conf > ATTACK_CONFIDENCE_THRESHOLD
                         and has_enough_samples)

    # Gate 2 — persistence: record what this window would call an attack as
    # (or None), and only let a MODEL-based call through once the SAME class
    # has repeated for CONSECUTIVE_WINDOWS_REQUIRED windows in a row. This is
    # what actually distinguishes "one quiet/sparse second that happens to
    # look like Slowloris" from "this has genuinely been trickling for
    # several seconds," which is the real definition of a low-and-slow attack.
    recent_attack_calls.append(pred_class if model_says_attack else None)
    persistence_confirmed = (
        len(recent_attack_calls) == CONSECUTIVE_WINDOWS_REQUIRED
        and all(c == pred_class for c in recent_attack_calls)
        and pred_class != BENIGN_LABEL
    )

    # Large-sample signals don't need persistence — a genuine volumetric
    # flood or a big statistical spike is already unambiguous in one window.
    is_volumetric = pkt_count > 1000
    is_statistical = z > 4.0

    # Gate 3 — novel-attack signal: the autoencoder flags this window as
    # abnormal (doesn't reconstruct like benign traffic) but the CNN still
    # confidently called it BENIGN. That combination is exactly what we
    # measured happening on an attack type the CNN was never trained on
    # (see evaluate_autoencoder.py, Part B): the CNN can't say "unknown," so
    # it's forced onto one of its 3 labels and, on genuinely novel traffic,
    # picks the wrong one with high confidence. The autoencoder has no such
    # constraint and catches it directly. Still requires MIN_PKTS_FOR_ATTACK_CALL
    # so a single sparse ping window can't trip this on reconstruction noise.
    is_novel_anomaly = is_anomalous and pred_class == BENIGN_LABEL and has_enough_samples

    if persistence_confirmed or is_volumetric or is_statistical or is_novel_anomaly:
        attacker_ip = Counter(src_ips).most_common(1)[0][0]
        # If the CNN didn't confidently name an attack type but volume/stats
        # tripped the trigger, still surface a best-guess label for the UI.
        if pred_class != BENIGN_LABEL:
            reported_type = pred_class
        elif is_novel_anomaly:
            # Anomalous per the autoencoder, but the CNN doesn't recognize
            # it as any of its known types either -- genuinely novel, not
            # just an unconfident volumetric call.
            reported_type = "UNKNOWN_ANOMALY (novel — unrecognized by CNN)"
        else:
            reported_type = "UNKNOWN_ANOMALY"
        trigger_reason = ("model+persistence" if persistence_confirmed else
                           "volume" if is_volumetric else
                           "statistical" if is_statistical else
                           "autoencoder novel-anomaly")
        print(f"\n[🚨 ATTACK DETECTED] Type: {reported_type} | Trigger: {trigger_reason} | "
              f"Confidence: {pred_conf*100:.2f}% | AE_error={recon_error:.4f} | z={z:+.2f} | Rate: {pkt_count} Pkt/s")

        newly_blocked = None
        block_reason = None
        if attacker_ip not in blocked_ips:
            allowed, reason = should_block_ip(attacker_ip, reported_type)
            block_reason = reason
            print(f"[🛡️ SPOOF CHECK] {reason}")
            if allowed:
                print(f"[📡 SIGNAL] Instructing Ryu to drop attacker: {attacker_ip}")
                try:
                    requests.post("http://127.0.0.1:8080/mitigate", json={"ip": attacker_ip})
                    blocked_ips.add(attacker_ip)
                    newly_blocked = attacker_ip
                except Exception as e:
                    print(f"[-] Ryu communication fault: {e}")
            else:
                print(f"[⏸️  BLOCK WITHHELD] {attacker_ip} not blocked — {reason}")

        was_locked_down = is_lockdown_active()
        arm_lockdown()   # always refresh the timer on a confirmed attack
        if not was_locked_down:
            print(f"[🔒 SYSTEM LOCKDOWN] Threat confirmed. Activating continuous RSA protection "
                  f"(auto-lifts after {LOCKDOWN_TIMEOUT_SECONDS}s with no further confirmed attacks)...")

        report_telemetry("ATTACK", pred_conf, attacker_ip=attacker_ip,
                          new_blocked_ip=newly_blocked, stat_zscore=z,
                          attack_type=reported_type, class_probs=class_probs,
                          block_reason=block_reason)
    else:
        if is_lockdown_active():
            print(f"[🛡️ CONTINUOUS RSA ACTIVE] Fabric stable ({pkt_count} Pkt/s). Gateway remains locked.")
            report_telemetry("LOCKDOWN", pred_conf, stat_zscore=z,
                              attack_type=BENIGN_LABEL, class_probs=class_probs)
        elif model_says_attack:
            # The model flagged this window as non-benign, but it hasn't
            # repeated for enough consecutive windows to act on yet — surface
            # that as its own state rather than silently calling it SAFE.
            print(f"[🟡 SUSPECT] {pred_class} ({pred_conf*100:.1f}%) — watching for persistence "
                  f"({len([c for c in recent_attack_calls if c == pred_class])}/{CONSECUTIVE_WINDOWS_REQUIRED} windows)")
            report_telemetry("SUSPECT", pred_conf, stat_zscore=z,
                              attack_type=pred_class, class_probs=class_probs)
        else:
            report_telemetry("SAFE", pred_conf, stat_zscore=z,
                              attack_type=BENIGN_LABEL, class_probs=class_probs)

if __name__ == '__main__':
    while True:
        main_loop()