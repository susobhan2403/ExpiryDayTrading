import json, hashlib, requests, os
from urllib.parse import urlencode

CONFIG = "settings.json"
with open(CONFIG, "r") as f:
    cfg = json.load(f)

API_KEY = cfg["KITE_API_KEY"]
API_SECRET = cfg["KITE_API_SECRET"]

login_url = f"https://kite.trade/connect/login?{urlencode({'api_key': API_KEY, 'v': 3})}"
print("\n1) Open this URL in your browser, login to Kite:")
print(login_url)
print("\n2) After login, copy the value of request_token from your browser address bar.\n")
request_token = input("Paste request_token: ").strip()

checksum = hashlib.sha256(f"{API_KEY}{request_token}{API_SECRET}".encode()).hexdigest()
r = requests.post(
    "https://api.kite.trade/session/token",
    data={"api_key": API_KEY, "request_token": request_token, "checksum": checksum},
    headers={"X-Kite-Version": "3"}
)
r.raise_for_status()
access_token = r.json()["data"]["access_token"]

with open(".kite_session.json", "w") as f:
    json.dump({"access_token": access_token}, f)

print("✅ Access token saved to .kite_session.json")
