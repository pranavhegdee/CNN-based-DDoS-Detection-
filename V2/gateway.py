import jwt
import os
import threading
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Try to load the RSA Public Key for signature validation
try:
    with open("public_key.pem", "rb") as f:
        PUBLIC_KEY = f.read()
except FileNotFoundError:
    print("[-] Error: public_key.pem not found. Generate it using OpenSSL first!")

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

@app.route('/service', methods=['GET', 'POST'])
def handle_request():
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
    
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        report_gateway_event("rsa", "denied", "Missing bearer token.")
        return jsonify({
            "error": "Access Denied. Strict RSA validation token required."
        }), 401
        
    token = auth_header.split(" ")[1]
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
        report_gateway_event("rsa", "denied", str(e))
        return jsonify({
            "error": f"Access Denied. Cryptographic signature verification failed: {str(e)}"
        }), 401

if __name__ == '__main__':
    # Running single-threaded to preserve application state stability
    app.run(host='0.0.0.0', port=5000, threaded=False)
