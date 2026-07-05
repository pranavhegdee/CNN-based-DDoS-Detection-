import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
num_samples = 1000  # 1000 normal, 1000 attack rows

print("[*] Engineering a high-fidelity network dataset...")

# --- 1. GENERATE NORMAL TRAFFIC SIGNATURES (Label: 0) ---
# Clean users pace packets out with low bytes and occasional small spikes
normal_pps = np.random.normal(loc=15, scale=5, size=num_samples).clip(0)
normal_bps = normal_pps * np.random.normal(loc=74, scale=10, size=num_samples).clip(64)
normal_syn = np.random.binomial(n=1, p=0.05, size=num_samples) # Low SYN flags
normal_udp = normal_pps * np.random.uniform(0.1, 0.3, size=num_samples)

df_normal = pd.DataFrame({
    'packets_per_sec': normal_pps,
    'bytes_per_sec': normal_bps,
    'syn_packets': normal_syn,
    'udp_packets': normal_udp,
    'label': 0
})

# --- 2. GENERATE DDOS FLOOD TRAFFIC SIGNATURES (Label: 1) ---
# Attackers saturate queues with massive packet frequencies and packet sizes
attack_pps = np.random.normal(loc=35000, scale=4000, size=num_samples).clip(15000)
attack_bps = attack_pps * np.random.normal(loc=1200, scale=50, size=num_samples) # Heavy 1200-byte packets
attack_syn = np.random.normal(loc=100, scale=20, size=num_samples).clip(0)
attack_udp = attack_pps * np.random.uniform(0.8, 0.95, size=num_samples) # Mostly UDP flood

df_attack = pd.DataFrame({
    'packets_per_sec': attack_pps,
    'bytes_per_sec': attack_bps,
    'syn_packets': attack_syn,
    'udp_packets': attack_udp,
    'label': 1
})

# Combine, shuffle, and save the dataset
dataset = pd.concat([df_normal, df_attack]).sample(frac=1).reset_index(drop=True)
dataset.to_csv("traffic_dataset_premium.csv", index=False)

print(f"[====>] Premium dataset generated successfully! Total rows: {len(dataset)}")
