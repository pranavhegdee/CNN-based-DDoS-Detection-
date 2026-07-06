import jwt
import time
import requests

try:
    # Load the Private Key (Must stay strictly on authorized client nodes)
    with open("private_key.pem", "rb") as f:
        PRIVATE_KEY = f.read()
except FileNotFoundError:
    print("[-] Error: private_key.pem missing on client. Run openssl command first.")
    exit(1)

def send_authenticated_request():
    # Construct a temporary time-locked authorization token
    payload = {
        "user": "legitimate_node_h2",
        "exp": time.time() + 60  # Token expires automatically after 60 seconds
    }
    
    # Asymmetrically sign the payload token with your Private Key
    token = jwt.encode(payload, PRIVATE_KEY, algorithm="RS256")
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get("http://10.0.0.1:5000/service", headers=headers)
        print(f"[*] Response Status Code: {response.status_code}")
        print(f"[*] Response Body: {response.json()}")
    except Exception as e:
        print(f"[-] Connection failed: {e}")

if __name__ == '__main__':
    send_authenticated_request()
