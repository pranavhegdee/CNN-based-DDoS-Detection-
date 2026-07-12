"""
mininet_timing_benchmark.py
------------------------------
Run this ON YOUR MININET SETUP (not in a sandbox) to measure the timing
numbers that genuinely require a live OVS switch + Ryu controller +
running detection engine -- these cannot be honestly estimated offline.

Usage:
    1. Start dashboard_server.py, cnn_guard1.py, and your Ryu controller
       as normal.
    2. Run this script from a Mininet host or the root namespace:
           python3 mininet_timing_benchmark.py --attack syn
       (or --attack udp)
    3. It launches the attack, polls the dashboard's /api/state endpoint
       once per second, and reports the wall-clock time from attack
       launch to the first ATTACK/LOCKDOWN state and first blocked_ips
       entry -- i.e. real, measured detection + mitigation latency.
"""

import argparse
import subprocess
import time
import requests

DASHBOARD_URL = "http://127.0.0.1:5050/api/state"


def poll_until(condition_fn, timeout=30, interval=0.25):
    start = time.time()
    while time.time() - start < timeout:
        try:
            state = requests.get(DASHBOARD_URL, timeout=1).json()
            result = condition_fn(state)
            if result:
                return time.time() - start, state
        except Exception:
            pass
        time.sleep(interval)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", choices=["syn", "udp"], required=True)
    ap.add_argument("--target", default="10.0.0.1")
    ap.add_argument("--pre-attack-baseline", type=int, default=5,
                     help="seconds to record dashboard state before launching, for a sanity baseline")
    args = ap.parse_args()

    print("[*] Recording pre-attack baseline...")
    baseline_start = time.time()
    try:
        baseline_state = requests.get(DASHBOARD_URL, timeout=1).json()
        print(f"    status={baseline_state.get('status')}  blocked_ips={baseline_state.get('blocked_ips')}")
    except Exception as e:
        print(f"    [!] Could not reach dashboard at {DASHBOARD_URL}: {e}")
        print("    Make sure dashboard_server.py is running first.")
        return
    time.sleep(args.pre_attack_baseline)

    if args.attack == "syn":
        cmd = ["hping3", "-S", "--flood", "--rand-source", "-p", "80", args.target]
    else:
        cmd = ["hping3", "--udp", "--flood", "--rand-source", "-p", "80", args.target]

    print(f"[*] Launching attack: {' '.join(cmd)}")
    t_launch = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    t_detect, state_at_detect = poll_until(
        lambda s: s.get("status") in ("ATTACK", "LOCKDOWN"), timeout=15
    )
    t_block, state_at_block = poll_until(
        lambda s: len(s.get("blocked_ips", [])) > 0, timeout=15
    )

    proc.terminate()
    try:
        proc.wait(timeout=3)
    except Exception:
        proc.kill()

    print("\n" + "=" * 55)
    print("RESULTS")
    print("=" * 55)
    if t_detect is not None:
        print(f"Time from attack launch to detected state (ATTACK/LOCKDOWN): {t_detect:.2f}s")
        print(f"  attack_type reported: {state_at_detect.get('attack_type')}")
    else:
        print("Detection state was NOT observed within timeout -- check cnn_guard1.py is running")
        print("and sniffing the correct interface.")

    if t_block is not None:
        print(f"Time from attack launch to first blocked_ips entry (VACL pushed): {t_block:.2f}s")
        print(f"  blocked_ips: {state_at_block.get('blocked_ips')}")
    else:
        print("No IP was blocked within timeout -- if this was a --rand-source flood, check whether")
        print("the spoof-verification logic withheld the block (see [SPOOF CHECK] logs) rather than")
        print("assuming mitigation failed outright.")

    print("\nNote: this measures wall-clock time INCLUDING the dashboard's own ~1s telemetry poll")
    print("interval, so treat this as an upper bound on true detection latency, not an exact figure.")
    print("For the precise OpenFlow rule-install time itself, cross-check with:")
    print("    sh ovs-ofctl dump-flows s1   (timestamp when the priority-10000 rule appears)")


if __name__ == "__main__":
    main()
