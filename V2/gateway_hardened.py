import jwt
import os
import re
import time
import threading
import requests
from collections import defaultdict, deque
from flask import Flask, request, jsonify
from rsa_mitigation_queue import enqueue as enqueue_mitigation

app = Flask(__name__)

# Try to load the RSA Public Key for signature validation
try:
    with open("public_key.pem", "rb") as f:
        PUBLIC_KEY = f.read()
except FileNotFoundError:
    print("[-] Error: public_key.pem not found. Generate it using OpenSSL first!")
    PUBLIC_KEY = None  # fail CLOSED below rather than crashing at import time

# ---- Dashboard reporting -----------------------------------------------
# Fire-and-forget so a missing/slow dashboard never delays a real request.
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://127.0.0.1:5050")

def report_gateway_event(mode, result, detail="", user=None):
    if not DASHBOARD_URL:
        return
    def _send():
        try:
            requests.post(f"{DASHBOARD_URL}/api/gateway_event", json={
                "mode": mode, "result": result, "detail": detail, "user": user,
            }, timeout=0.5)
        except Exception:
            pass
    threading.Thread(target=_send, daemon=True).start()
# -------------------------------------------------------------------------

# ---- RSA token-exhaustion defense ---------------------------------------
# A flood of invalid-signature tokens costs real CPU (RSA verification)
# even when every connection is legitimate TCP and every request is
# well-formed HTTP -- nothing about it looks like a network-layer flood,
# so the CNN/autoencoder pipeline watching packet stats has no way to see
# this. Three defenses, cheapest first:
#
#  1. Format pre-check BEFORE calling jwt.decode(): a JWT is always three
#     base64url segments separated by dots. Garbage that doesn't even have
#     that shape gets rejected on a regex match, never touching the RSA
#     math at all. This is the cheap 99% case for a naive flood.
#  2. Per-source sliding-window failure tracking: even a well-shaped-but-
#     wrong-signature token DOES cost a real RSA verification, so track
#     failures per IP over a short window and escalate once a source
#     crosses a threshold -- this is the expensive case a format check
#     can't stop by itself.
#  3. Escalation reuses your EXISTING Ryu mitigation path: once a source
#     is confirmed abusive, push the same hardware-speed VACL drop rule
#     cnn_guard1.py already uses for network-layer attacks, plus report a
#     distinct RSA_TOKEN_EXHAUSTION event to the dashboard so it's visible
#     as its own attack type, not lumped in with SYN/UDP floods.
JWT_SHAPE_RE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")

FAILURE_WINDOW_SECONDS = 10
FAILURE_THRESHOLD = 15          # more than this many bad-signature verifies
                                 # from one IP within the window -> mitigate

_failure_lock = threading.Lock()
_failures_by_src = defaultdict(deque)   # ip -> deque of failure timestamps
_mitigated_ips = set()

def _record_failure_and_maybe_mitigate(src_ip):
    """Returns True if this call just triggered a fresh mitigation."""
    now = time.time()
    with _failure_lock:
        dq = _failures_by_src[src_ip]
        dq.append(now)
        while dq and now - dq[0] > FAILURE_WINDOW_SECONDS:
            dq.popleft()
        count = len(dq)
        if count > FAILURE_THRESHOLD and src_ip not in _mitigated_ips:
            _mitigated_ips.add(src_ip)
            return True, count
        return False, count

def _mitigate(src_ip, count):
    # gateway.py runs inside a Mininet HOST's network namespace, so it
    # cannot reach the Ryu controller's REST API directly -- that lives in
    # a different namespace and "127.0.0.1:8080" from here would just be
    # this host's own loopback, not the controller's. Hand the IP off via
    # the shared-file queue instead; cnn_guard1.py, which already has a
    # WORKING path to Ryu, polls this queue and performs the actual
    # hardware block using its existing, already-tested code path.
    reason = f"RSA_TOKEN_EXHAUSTION: {count} bad-signature verifications in {FAILURE_WINDOW_SECONDS}s"
    print(f"[🚨 RSA TOKEN EXHAUSTION] {src_ip} sent {count} bad-signature tokens in "
          f"{FAILURE_WINDOW_SECONDS}s -- handing off to cnn_guard1.py for hardware block.")
    enqueue_mitigation(src_ip, count, reason)
    report_gateway_event("rsa", "mitigation_requested", detail=reason, user=src_ip)
# -------------------------------------------------------------------------

@app.route('/service', methods=['GET', 'POST'])
def handle_request():
    src_ip = request.remote_addr

    # Already-mitigated sources shouldn't even reach RSA verification again
    # (belt-and-suspenders alongside the actual OpenFlow drop rule, in case
    # traffic reaches the app before the switch rule takes effect).
    if src_ip in _mitigated_ips:
        return jsonify({"error": "Source blocked due to prior abuse."}), 403

    # Dynamically check if the AI engine has created the lockdown flag file
    lockdown_armed = os.path.exists("/tmp/lockdown.txt")

    # Tier 1: Normal mode (No active or historical attack detected)
    if not lockdown_armed:
        report_gateway_event("normal", "success", "Standard open channel transaction complete.")
        return jsonify({
            "status": "Success",
            "data": "Standard open channel transaction complete."
        }), 200

    # Tier 2: Continuous Cryptographic RSA Mode
    print("[🔒 RSA MODE ACTIVE] Shielding backend asset. Validating token signature...")

    if PUBLIC_KEY is None:
        # Fail CLOSED: no key material means we cannot verify anything, so
        # deny rather than silently falling back to open access.
        report_gateway_event("rsa", "denied", "Server misconfiguration: public_key.pem missing.")
        return jsonify({"error": "Access Denied. Server key material unavailable."}), 503

    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        report_gateway_event("rsa", "denied", "Missing bearer token.")
        return jsonify({
            "error": "Access Denied. Strict RSA validation token required."
        }), 401

    token = auth_header.split(" ")[1]

    # Cheap pre-check BEFORE any RSA math: reject obviously-malformed
    # tokens on a regex match. This is what stops a naive high-rate flood
    # of garbage strings from costing anything beyond string comparison.
    if not JWT_SHAPE_RE.match(token):
        triggered, count = _record_failure_and_maybe_mitigate(src_ip)
        if triggered:
            _mitigate(src_ip, count)
        report_gateway_event("rsa", "denied", "Malformed token (failed shape pre-check).")
        return jsonify({"error": "Access Denied. Malformed token."}), 401

    try:
        # Cryptographically validate the token signature using the Public Key
        decoded_payload = jwt.decode(token, PUBLIC_KEY, algorithms=["RS256"])
        print(f"[💚 RSA VERIFIED] Legitimate access by client node: {decoded_payload.get('user')}")
        report_gateway_event("rsa", "success", "Verified via RS256 keypair.", user=decoded_payload.get('user'))
        return jsonify({
            "status": "Success",
            "crypto_validation": "Verified via RS256 Keypair"
        }), 200
    except Exception as e:
        # Well-shaped but cryptographically invalid: this DID cost a real
        # RSA verification, so it's the case the per-IP counter exists for.
        triggered, count = _record_failure_and_maybe_mitigate(src_ip)
        if triggered:
            _mitigate(src_ip, count)
        report_gateway_event("rsa", "denied", str(e))
        return jsonify({
            "error": f"Access Denied. Cryptographic signature verification failed: {str(e)}"
        }), 401

if __name__ == '__main__':
    # threaded=True: the ORIGINAL single-threaded server (threaded=False) is
    # itself part of the vulnerability -- it processes exactly one request
    # at a time regardless of how many connections queue up, so even a
    # moderate-rate flood of ANY requests (valid or not) starves legitimate
    # clients simply by occupying the one worker. Threading lets concurrent
    # legitimate requests actually get served while abusive ones are being
    # rejected/rate-limited. For a real deployment beyond a Mininet demo,
    # a proper WSGI server (gunicorn with multiple workers) is the further
    # step beyond what Flask's dev server can offer.
    app.run(host='0.0.0.0', port=5000, threaded=True)