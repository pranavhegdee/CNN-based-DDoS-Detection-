import jwt
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# Try to load the RSA Public Key for signature validation
try:
    with open("public_key.pem", "rb") as f:
        PUBLIC_KEY = f.read()
except FileNotFoundError:
    print("[-] Error: public_key.pem not found. Generate it using OpenSSL first!")

@app.route('/service', methods=['GET', 'POST'])
def handle_request():
    # Dynamically check if the AI engine has created the lockdown flag file
    lockdown_armed = os.path.exists("/tmp/lockdown.txt")
    
    # Tier 1: Normal mode (No active or historical attack detected)
    if not lockdown_armed:
        return jsonify({
            "status": "Success", 
            "data": "Standard open channel transaction complete."
        }), 200
        
    # Tier 2: Continuous Cryptographic RSA Mode
    print("[🔒 RSA MODE ACTIVE] Shielding backend asset. Validating token signature...")
    
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({
            "error": "Access Denied. Strict RSA validation token required."
        }), 401
        
    token = auth_header.split(" ")[1]
    try:
        # Cryptographically validate the token signature using the Public Key
        decoded_payload = jwt.decode(token, PUBLIC_KEY, algorithms=["RS256"])
        print(f"[💚 RSA VERIFIED] Legitimate access by client node: {decoded_payload.get('user')}")
        return jsonify({
            "status": "Success", 
            "crypto_validation": "Verified via RS256 Keypair"
        }), 200
    except Exception as e:
        return jsonify({
            "error": f"Access Denied. Cryptographic signature verification failed: {str(e)}"
        }), 401

if __name__ == '__main__':
    # Running single-threaded to preserve application state stability
    app.run(host='0.0.0.0', port=5000, threaded=False)
