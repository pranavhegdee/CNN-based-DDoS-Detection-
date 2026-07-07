import time
import pickle
import os
import requests
import numpy as np
import tensorflow as tf
from scapy.all import sniff, IP, TCP
from collections import Counter

print("[*] Initializing multi-class deep learning defense architectures...")
model = tf.keras.models.load_model("traffic_cnn_model_premium.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ---- Dashboard reporting -----------------------------------------------
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://127.0.0.1:5050")

def report_telemetry(status, prediction, attack_type="BENIGN", attacker_ip=None, new_blocked_ip=None):
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
            "attack_type": attack_type,
            "status": status,
            "lockdown": os.path.exists("/tmp/lockdown.txt"),
            "attacker_ip": attacker_ip,
            "new_blocked_ip": new_blocked_ip,
            "blocked_ips": list(blocked_ips),
        }, timeout=0.5)
    except Exception:
        pass
# -------------------------------------------------------------------------

# Global operational metric counters
pkt_count = fwd_pkts = bwd_pkts = fwd_len = bwd_len = max_fwd_len = min_fwd_len = max_bwd_len = min_bwd_len = 0
tcp_syn_flags = tcp_ack_flags = tcp_rst_flags = tcp_psh_flags = tcp_fin_flags = tcp_urg_flags = 0
icmp_pkts = tcp_pkts = udp_pkts = 0
packet_sizes = []
src_ips = []
dest_ports = []
blocked_ips = set()

# --- Automated Cooldown Tracker ---
safe_cycles = 0  # Tracks sequential seconds of clean traffic
COOLDOWN_TIMEOUT = 15  # Number of stable seconds required to remove RSA lockdown

def process_packet(packet):
    global pkt_count, fwd_pkts, bwd_pkts, fwd_len, bwd_len, max_fwd_len, min_fwd_len, max_bwd_len, min_bwd_len
    global tcp_syn_flags, tcp_ack_flags, tcp_rst_flags, tcp_psh_flags, tcp_fin_flags, tcp_urg_flags
    global icmp_pkts, tcp_pkts, udp_pkts, packet_sizes, src_ips, dest_ports

    if packet.haslayer(IP):
        pkt_count += 1
        p_len = len(packet)
        packet_sizes.append(p_len)
        src_ips.append(packet[IP].src)
        
        # --- Protocol Identification Block ---
        if packet.haslayer(TCP):
            tcp_pkts += 1
            dest_ports.append(packet[TCP].dport)
            flags = packet[TCP].flags
            if flags & 0x02: tcp_syn_flags += 1
            if flags & 0x10: tcp_ack_flags += 1
            if flags & 0x04: tcp_rst_flags += 1
            if flags & 0x08: tcp_psh_flags += 1
            if flags & 0x01: tcp_fin_flags += 1
            if flags & 0x20: tcp_urg_flags += 1
        elif packet[IP].proto == 17:  # UDP
            udp_pkts += 1
        elif packet[IP].proto == 1:   # ICMP (Ping)
            icmp_pkts += 1

        # Route directional flow features
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

def execute_mitigation_playbook(attack_class, attacker_ip, confidence):
    """Dynamically executes specific defensive maneuvers per attack signature."""
    global blocked_ips, safe_cycles
    
    # An attack is active, instantly reset the cooldown timer back to zero
    safe_cycles = 0
    
    print(f"\n[🚨 MULTI-CLASS ALERT] Threat: {attack_class} | Confidence: {confidence*100:.2f}%")
    newly_blocked = None
    
    # Maneuver 1: Network-Layer Hardware Drop Rule via OpenFlow
    if attack_class in ["UDP_FLOOD", "SYN_FLOOD", "ICMP_FLOOD", "DNS_AMPLIFICATION", "PORT_SCAN"]:
        if attacker_ip not in blocked_ips and attacker_ip != "10.0.0.1":
            print(f"[📡 SDN SIGNAL] Ordering Ryu Controller to drop hardware traffic from: {attacker_ip}")
            try:
                requests.post("http://127.0.0.1:8080/mitigate", json={"ip": attacker_ip})
                blocked_ips.add(attacker_ip)
                newly_blocked = attacker_ip
            except Exception as e:
                print(f"[-] Ryu communication fault: {e}")

    # Maneuver 2: Application-Layer Asymmetric RSA Authentication Checkpoint Activation
    if attack_class in ["UDP_FLOOD", "HTTP_FLOOD", "SLOWLORIS", "DNS_AMPLIFICATION"]:
        if not os.path.exists("/tmp/lockdown.txt"):
            print("[🔒 APP LOCKDOWN] Raising continuous RSA token checkpoint on Web Gateway...")
            try:
                with open("/tmp/lockdown.txt", "w") as f:
                    f.write("ARMED")
            except Exception as e:
                print(f"[-] Failed to write lockdown flag: {e}")
                
    return newly_blocked

def main_loop():
    global pkt_count, fwd_pkts, bwd_pkts, fwd_len, bwd_len, max_fwd_len, min_fwd_len, max_bwd_len, min_bwd_len
    global tcp_syn_flags, tcp_ack_flags, tcp_rst_flags, tcp_psh_flags, tcp_fin_flags, tcp_urg_flags
    global icmp_pkts, tcp_pkts, udp_pkts, packet_sizes, src_ips, dest_ports, safe_cycles
    
    # Clear sliding window tracking states
    pkt_count = fwd_pkts = bwd_pkts = fwd_len = bwd_len = max_fwd_len = min_fwd_len = max_bwd_len = min_bwd_len = 0
    tcp_syn_flags = tcp_ack_flags = tcp_rst_flags = tcp_psh_flags = tcp_fin_flags = tcp_urg_flags = 0
    icmp_pkts = tcp_pkts = udp_pkts = 0
    packet_sizes = [0]
    src_ips = []
    dest_ports = []
    
    # 1. Listen to mirror interface link
    sniff(iface="s1-eth1", timeout=1, prn=process_packet, store=False)
    if pkt_count == 0: 
        # If absolute silence is observed while in lockdown, count it towards the cooldown window
        if os.path.exists("/tmp/lockdown.txt"):
            safe_cycles += 1
            if safe_cycles >= COOLDOWN_TIMEOUT:
                print("[🔓 AUTO COOLDOWN] Network silent. Removing RSA application lockdown.")
                if os.path.exists("/tmp/lockdown.txt"): os.remove("/tmp/lockdown.txt")
                safe_cycles = 0
            else:
                print(f"[🛡️ LOCKDOWN ACTIVE] Fabric silent. Cooldown: {safe_cycles}/{COOLDOWN_TIMEOUT}s")
        else:
            safe_cycles = 0
            
        report_telemetry("LOCKDOWN" if os.path.exists("/tmp/lockdown.txt") else "SAFE", 0.0, "BENIGN")
        return

    # 2. Re-compute runtime window metrics
    flow_duration = 1.0
    avg_packet_size = np.mean(packet_sizes)
    
    primary_protocol = 1.0  # Default to TCP
    if icmp_pkts > tcp_pkts and icmp_pkts > udp_pkts:
        primary_protocol = 3.0
    elif udp_pkts > tcp_pkts:
        primary_protocol = 2.0
    
    # Build raw feature vectors
    raw_vector = np.array([[
        flow_duration, fwd_pkts, bwd_pkts, fwd_len, bwd_len,
        max_fwd_len, min_fwd_len, max_bwd_len, min_bwd_len,
        avg_packet_size, tcp_syn_flags, tcp_ack_flags,
        tcp_rst_flags, tcp_psh_flags, tcp_fin_flags, tcp_urg_flags,
        primary_protocol
    ]])
    
    # 3. Shape validation matching scale configuration
    num_features_expected = scaler.n_features_in_
    if raw_vector.shape[1] < num_features_expected:
        padding = np.zeros((1, num_features_expected - raw_vector.shape[1]))
        features_ready = np.hstack((raw_vector, padding))
    else:
        features_ready = raw_vector[:, :num_features_expected]
        
    # 4. Reshape to 8x8 matrix image shape
    scaled_features = scaler.transform(features_ready)
    if scaled_features.shape[1] < 64:
        image_padding = np.zeros((1, 64 - scaled_features.shape[1]))
        scaled_features = np.hstack((scaled_features, image_padding))
    else:
        scaled_features = scaled_features[:, :64]
        
    traffic_image = scaled_features.reshape(-1, 8, 8, 1)
    
    # 5. Model Prediction Evaluate
    probabilities = model.predict(traffic_image, verbose=0)[0]
    predicted_class_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_class_idx]
    attack_class = label_encoder.inverse_transform([predicted_class_idx])[0]
    
    # --- CRITICAL SAFETY OVERRIDE GUARDRAIL ---
    if attack_class == "SLOWLORIS" and tcp_pkts == 0:
        attack_class = "BENIGN"
        confidence = 0.0
    
    # 6. Fallback Heuristics Check
    is_fallback_attack = False
    if pkt_count > 500 and primary_protocol == 2.0:
        attack_class = "UDP_FLOOD"
        confidence = 1.0
        is_fallback_attack = True
    elif pkt_count > 500 and primary_protocol == 3.0:
        attack_class = "ICMP_FLOOD"
        confidence = 1.0
        is_fallback_attack = True

    # 7. Action Matrix Routing Logic
    if (attack_class != "BENIGN" and confidence > 0.45) or is_fallback_attack:
        attacker_ip = Counter(src_ips).most_common(1)[0][0] if src_ips else "0.0.0.0"
        new_blocked = execute_mitigation_playbook(attack_class, attacker_ip, confidence)
        report_telemetry("ATTACK", confidence, attack_type=attack_class, attacker_ip=attacker_ip, new_blocked_ip=new_blocked)
    else:
        # --- TRAFFIC IS BENIGN: RUN COOLDOWN EVALUATION ---
        if os.path.exists("/tmp/lockdown.txt"):
            safe_cycles += 1
            print(f"[⏳ COOLDOWN STAGE] Traffic safe ({pkt_count} Pkt/s). Progress: {safe_cycles}/{COOLDOWN_TIMEOUT} seconds.")
            
            # Timeout reached — automatically remove the lock flag
            if safe_cycles >= COOLDOWN_TIMEOUT:
                print("[🔓 AUTO COOLDOWN COMPLETE] Network verified stable. Removing RSA web lockdown.")
                if os.path.exists("/tmp/lockdown.txt"): os.remove("/tmp/lockdown.txt")
                safe_cycles = 0
                report_telemetry("SAFE", confidence, attack_type="BENIGN")
            else:
                report_telemetry("LOCKDOWN", confidence, attack_type="BENIGN")
        else:
            safe_cycles = 0  # Maintain zero state if already unshielded
            print(f"[💚 BENIGN] Network fabric operating normally: {pkt_count} Pkt/s")
            report_telemetry("SAFE", confidence, attack_type="BENIGN")

if __name__ == '__main__':
    while True:
        main_loop()
