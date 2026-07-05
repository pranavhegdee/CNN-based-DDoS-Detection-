import time
import csv
from scapy.all import sniff, IP, TCP, UDP

# Output dataset file
CSV_FILE = "traffic_dataset.csv"

# Real-time traffic counters
packet_count = 0
byte_count = 0
syn_count = 0
udp_count = 0

def packet_callback(packet):
    global packet_count, byte_count, syn_count, udp_count
    if packet.haslayer(IP):
        packet_count += 1
        byte_count += len(packet)
        
        if packet.haslayer(TCP) and packet[TCP].flags == 'S':
            syn_count += 1
        elif packet.haslayer(UDP):
            udp_count += 1

# Initialize CSV file with headers
with open(CSV_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["packets_per_sec", "bytes_per_sec", "syn_packets", "udp_packets", "label"])

print("[*] Starting data collection window. Capture 'Normal' traffic first...")
# Change label to 0 for normal, 1 for attack when running your hping3 flood
CURRENT_LABEL = 1  

while True:
    try:
        # Sniff packets for exactly 1 second
        sniff(iface="h1-eth0", prn=packet_callback, timeout=1, store=False)
        
        # Append the 1-second interval metrics to our dataset
        with open(CSV_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([packet_count, byte_count, syn_count, udp_count, CURRENT_LABEL])
        
        print(f"[+] Recorded Window - Pkts: {packet_count} | Label: {CURRENT_LABEL}")
        
        # Reset counters for the next second
        packet_count = byte_count = syn_count = udp_count = 0
    except KeyboardInterrupt:
        print("\n[*] Data collection stopped.")
        break
