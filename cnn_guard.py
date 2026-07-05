
import time
import pickle
import requests
import numpy as np
import tensorflow as tf
from scapy.all import sniff, IP, TCP
from collections import Counter

print("[*] Loading 2D CNN Network Model Components...")
model = tf.keras.models.load_model("cic2019_2d_cnn.h5")
with open("scaler_cic2019.pkl", "rb") as f:
    scaler = pickle.load(f)

# Global metrics tracking window variables
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
        
        # Calculate true feature flow direction distributions
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
            
        if packet.haslayer(TCP) and 'U' in str(packet[TCP].flags):
            urg_flags += 1

def main_loop():
    global pkt_count, fwd_pkts, bwd_pkts, fwd_len, bwd_len, max_fwd_len, min_fwd_len, max_bwd_len, min_bwd_len, urg_flags, packet_sizes, src_ips
    
    # Reset tracking state arrays for the current 1-second monitoring slice
    pkt_count = fwd_pkts = bwd_pkts = fwd_len = bwd_len = max_fwd_len = min_fwd_len = max_bwd_len = min_bwd_len = urg_flags = 0
    packet_sizes = [0]
    src_ips = []
    
    # Sniff the central Mininet switch link interface directly for exactly 1 second
    sniff(iface="s1-eth1", timeout=1, prn=process_packet, store=False)
    
    if pkt_count == 0: 
        return

    # --- Accurate 16 Feature Extraction Loop ---
    flow_duration = 1.0
    flow_bytes_sec = (fwd_len + bwd_len) / flow_duration
    flow_pkts_sec = pkt_count / flow_duration
    fwd_pkts_sec = fwd_pkts / flow_duration
    bwd_pkts_sec = bwd_pkts / flow_duration
    down_up_ratio = bwd_pkts / (fwd_pkts + 1e-5)
    avg_packet_size = np.mean(packet_sizes)
    
    # Compile raw array mirroring training dataset distributions exactly
    raw_features = np.array([[
        flow_duration, fwd_pkts, bwd_pkts, fwd_len, bwd_len,
        max_fwd_len, min_fwd_len, max_bwd_len, min_bwd_len,
        flow_bytes_sec, flow_pkts_sec, fwd_pkts_sec, bwd_pkts_sec,
        urg_flags, down_up_ratio, avg_packet_size
    ]])
    
    # Normalization scaler pass and 2D image structural matrix transformation
    scaled_features = scaler.transform(raw_features)
    traffic_image = scaled_features.reshape(-1, 4, 4, 1)
    
    # Run CNN inference pass
    prediction = model.predict(traffic_image, verbose=0)[0][0]
    
    if prediction > 0.5:
        # Isolate attacker via majority voting count
        attacker_ip = Counter(src_ips).most_common(1)[0][0]
        print(f"[🚨 ATTACK DETECTED] Anomaly Signature: {prediction*100:.2f}% | Vol: {pkt_count} Pkt/s")
        
        if attacker_ip not in blocked_ips and attacker_ip != "10.0.0.1":
            print(f"[📡 ALARM] Sending VACL webhook to Ryu Controller for IP: {attacker_ip}")
            try:
                r = requests.post("http://127.0.0.1:8080/mitigate", json={"ip": attacker_ip})
                if r.status_code == 200:
                    blocked_ips.add(attacker_ip)
            except Exception as e:
                print(f"[-] Failed to communicate with Ryu REST server: {e}")
    else:
        print(f"[💚 SAFE] Monitoring fabric interface... Net load: {pkt_count} Pkt/s")

# Continuous network execution loop
while True:
    main_loop()
