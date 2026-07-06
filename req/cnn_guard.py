import time
import pickle
import os
import requests
import numpy as np
import tensorflow as tf
from scapy.all import sniff, IP
from collections import Counter

print("[*] Initializing deep learning architectures...")
model = tf.keras.models.load_model("traffic_cnn_model_premium.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---- Dashboard reporting -----------------------------------------------
# Set DASHBOARD_URL="" to disable reporting without touching the rest of
# the file. Runs on a short timeout so a missing/slow dashboard never
# stalls the detection loop.
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://127.0.0.1:5050")

def report_telemetry(status, prediction, attacker_ip=None, new_blocked_ip=None):
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
            "lockdown": os.path.exists("/tmp/lockdown.txt"),
            "attacker_ip": attacker_ip,
            "new_blocked_ip": new_blocked_ip,
            "blocked_ips": list(blocked_ips),
        }, timeout=0.5)
    except Exception:
        pass  # dashboard offline — detection/mitigation must keep running regardless
# -------------------------------------------------------------------------

# Global metric counters
pkt_count = fwd_pkts = bwd_pkts = fwd_len = bwd_len = max_fwd_len = min_fwd_len = max_bwd_len = min_bwd_len = 0
urg_flags = 0
packet_sizes = []
src_ips = []
blocked_ips = set()

def process_packet(packet):
    global pkt_count, fwd_pkts, bwd_pkts, fwd_len, bwd_len, max_fwd_len, min_fwd_len, max_bwd_len, min_bwd_len, urg_flags, packet_sizes, src_ips
    if packet.haslayer(IP):
        pkt_count += 1
        p_len = len(packet)
        packet_sizes.append(p_len)
        src_ips.append(packet[IP].src)
        
        if packet[IP].dst == "10.0.0.1":
            fwd_pkts += 1
            fwd_len += p_len
            if p_len > max_fwd_len: max_fwd_len = p_len
            if min_fwd_len == 0 or p_len < min_fwd_len: min_fwd_len = p_len
        else:
            bwd_pkts += 1
            bwd_len += p_len
            if p_len > max_bwd_len: max_bwd_len = p_len
            if min_bwd_len == 0 or p_len < min_bwd_len: min_bwd_len = p_len

def main_loop():
    global pkt_count, fwd_pkts, bwd_pkts, fwd_len, bwd_len, max_fwd_len, min_fwd_len, max_bwd_len, min_bwd_len, urg_flags, packet_sizes, src_ips
    
    # Reset tracking arrays for the new 1-second window
    pkt_count = fwd_pkts = bwd_pkts = fwd_len = bwd_len = max_fwd_len = min_fwd_len = max_bwd_len = min_bwd_len = urg_flags = 0
    packet_sizes = [0]
    src_ips = []
    
    # Capture live metrics from the virtual switch link mirror
    sniff(iface="s1-eth1", timeout=1, prn=process_packet, store=False)
    if pkt_count == 0:
        report_telemetry("LOCKDOWN" if os.path.exists("/tmp/lockdown.txt") else "SAFE", 0.0)
        return

    # Extract foundational features
    flow_duration = 1.0
    avg_packet_size = np.mean(packet_sizes)
    
    # --- MAP TO STANDARD CICIDS2019 COMPATIBLE INDICES ---
    raw_features_64 = np.zeros((1, 64))
    
    raw_features_64[0, 0] = flow_duration      # Flow Duration
    raw_features_64[0, 1] = fwd_pkts          # Total Fwd Packets
    raw_features_64[0, 2] = bwd_pkts          # Total Bwd Packets
    raw_features_64[0, 3] = fwd_len           # Total Length of Fwd Packets
    raw_features_64[0, 4] = bwd_len           # Total Length of Bwd Packets
    raw_features_64[0, 5] = max_fwd_len       # Fwd Packet Length Max
    raw_features_64[0, 6] = min_fwd_len       # Fwd Packet Length Min
    raw_features_64[0, 7] = max_bwd_len       # Bwd Packet Length Max
    raw_features_64[0, 8] = min_bwd_len       # Bwd Packet Length Min
    
    # Derived rates
    raw_features_64[0, 9] = (fwd_len + bwd_len) / flow_duration  # Flow Bytes/s
    raw_features_64[0, 10] = pkt_count / flow_duration           # Flow Packets/s
    raw_features_64[0, 40] = avg_packet_size                      # Avg Packet Size
    
    # 1. Dual-Compatibility Scale Verification Block
    try:
        # Check if scaler expects the full 64 feature layout
        scaled_features = scaler.transform(raw_features_64)
    except ValueError:
        # Adaptive Fallback: Scaler expects 4 features only
        features_for_scaler = raw_features_64[:, :4]
        scaled_4 = scaler.transform(features_for_scaler)
        scaled_features = np.zeros((1, 64))
        scaled_features[:, :4] = scaled_4
    
    # 2. Reshape to the exact 8x8 matrix the CNN expects
    traffic_image = scaled_features.reshape(-1, 8, 8, 1)
    prediction = model.predict(traffic_image, verbose=0)[0][0]
    
    # 3. Anomaly Trigger: Activate on model confidence OR stark volumetric anomalies (>1000 Pkt/s)
    if prediction > 0.3 or pkt_count > 1000:
        attacker_ip = Counter(src_ips).most_common(1)[0][0]
        print(f"\n[🚨 ATTACK DETECTED] Engine Alert | Model Score: {prediction*100:.2f}% | Rate: {pkt_count} Pkt/s")
        
        newly_blocked = None
        # Action A: Push OpenFlow rule to Ryu Controller
        if attacker_ip not in blocked_ips and attacker_ip != "10.0.0.1":
            print(f"[📡 SIGNAL] Instructing Ryu to drop attacker: {attacker_ip}")
            try:
                requests.post("http://127.0.0.1:8080/mitigate", json={"ip": attacker_ip})
                blocked_ips.add(attacker_ip)
                newly_blocked = attacker_ip
            except Exception as e:
                print(f"[-] Ryu communication fault: {e}")
                
        # Action B: Create continuous RSA validation checkpoint flag
        if not os.path.exists("/tmp/lockdown.txt"):
            print("[🔒 SYSTEM LOCKDOWN] Threat confirmed. Activating continuous RSA protection...")
            try:
                with open("/tmp/lockdown.txt", "w") as f:
                    f.write("ARMED")
            except Exception as e:
                print(f"[-] Failed to write lockdown flag: {e}")

        report_telemetry("ATTACK", prediction, attacker_ip=attacker_ip, new_blocked_ip=newly_blocked)
    else:
        if os.path.exists("/tmp/lockdown.txt"):
            print(f"[🛡️ CONTINUOUS RSA ACTIVE] Fabric stable ({pkt_count} Pkt/s). Gateway remains locked.")
            report_telemetry("LOCKDOWN", prediction)
        else:
            print(f"[💚 SAFE] Fabric load stable: {pkt_count} Pkt/s")
            report_telemetry("SAFE", prediction)

if __name__ == '__main__':
    while True:
        main_loop()
