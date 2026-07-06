import time
import pickle
import numpy as np
import tensorflow as tf
from scapy.all import sniff, IP, TCP, UDP

# 1. Load the trained 2D CNN model and the scaler weights
print("[*] Booting up Live 2D-CNN Traffic Guard...")
model = tf.keras.models.load_model("cic2019_2d_cnn.h5")

with open("scaler_cic2019.pkl", "rb") as f:
    scaler = pickle.load(f)

# Real-time traffic tracking variables
pkt_count = fwd_pkts = bwd_pkts = fwd_len = bwd_len = max_fwd_len = min_fwd_len = max_bwd_len = min_bwd_len = 0
urg_flags = 0
packet_sizes = []
start_time = time.time()

def process_packet(packet):
    global pkt_count, fwd_pkts, bwd_pkts, fwd_len, bwd_len, max_fwd_len, min_fwd_len, max_bwd_len, min_bwd_len, urg_flags, packet_sizes
    
    if packet.haslayer(IP):
        pkt_count += 1
        p_len = len(packet)
        packet_sizes.append(p_len)
        
        # Determine direction relative to protected host h1 (10.0.0.1)
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

def run_detection_window():
    global pkt_count, fwd_pkts, bwd_pkts, fwd_len, bwd_len, max_fwd_len, min_fwd_len, max_bwd_len, min_bwd_len, urg_flags, packet_sizes
    
    # Reset window timers
    pkt_count = fwd_pkts = bwd_pkts = fwd_len = bwd_len = max_fwd_len = min_fwd_len = max_bwd_len = min_bwd_len = urg_flags = 0
    packet_sizes = [0]
    
    # Sniff traffic on h1's interface for exactly 1 second
    sniff(iface="s1-eth1", prn=process_packet, timeout=1, store=False)
    
    # Calculate derived interaction features matching the 16 training metrics
    flow_duration = 1.0  # 1-second processing window
    flow_bytes_sec = (fwd_len + bwd_len) / flow_duration
    flow_pkts_sec = pkt_count / flow_duration
    fwd_pkts_sec = fwd_pkts / flow_duration
    bwd_pkts_sec = bwd_pkts / flow_duration
    down_up_ratio = bwd_pkts / (fwd_pkts + 1e-5)
    avg_packet_size = np.mean(packet_sizes) if packet_sizes else 0
    
    # Construct the array in the exact feature sequence used during training
    raw_features = np.array([[
        flow_duration, fwd_pkts, bwd_pkts, fwd_len, bwd_len,
        max_fwd_len, min_fwd_len, max_bwd_len, min_bwd_len,
        flow_bytes_sec, flow_pkts_sec, fwd_pkts_sec, bwd_pkts_sec,
        urg_flags, down_up_ratio, avg_packet_size
    ]])
    
    # Normalize via our specific fitted scaler limits
    scaled_features = scaler.transform(raw_features)
    
    # Reshape metrics into the 2D matrix structure expected by Conv2D
    traffic_image = scaled_features.reshape(-1, 4, 4, 1)
    
    # Run prediction inference
    prediction = model.predict(traffic_image, verbose=0)[0][0]
    
    if prediction > 0.5:
        print(f"[🚨 MITIGATE 🚨] DDoS Threat Identified! Probability: {prediction*100:.2f}% | Packets/sec: {pkt_count}")
    else:
        print(f"[💚 SAFE 💚] Regular traffic signature verified. Probability: {prediction*100:.2f}% | Packets/sec: {pkt_count}")

print("[+] Active Protection Loop Deployed. Awaiting traffic streams...")
while True:
    run_detection_window()
