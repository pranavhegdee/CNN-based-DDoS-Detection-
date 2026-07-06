"""
SDN Guard Dashboard Server
--------------------------
A lightweight aggregation + UI server for the CNN traffic-guard / Ryu
mitigation pipeline. It does not touch your detection or mitigation logic —
cnn_guard.py and gateway.py just POST small JSON updates here, and this
process keeps a short rolling history in memory and serves a live web UI.

Run this BEFORE (or alongside) cnn_guard.py, gateway.py and the Ryu
controller. Default port is 5050 so it doesn't collide with gateway.py
(5000) or the Ryu WSGI API (8080).

    python3 dashboard_server.py

Then open http://127.0.0.1:5050 in a browser.
"""

import time
import threading
from collections import deque

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
LOCK = threading.Lock()

HISTORY_LEN = 120     # ~2 minutes at 1 sample/sec
LOG_LEN = 60

state = {
    "status": "SAFE",              # SAFE | ATTACK | LOCKDOWN
    "pkt_count": 0,
    "fwd_pkts": 0,
    "bwd_pkts": 0,
    "fwd_len": 0,
    "bwd_len": 0,
    "prediction": 0.0,
    "attacker_ip": None,
    "blocked_ips": [],
    "lockdown": False,
    "last_update": None,
    "connected": False,
}

history = deque(maxlen=HISTORY_LEN)        # traffic + score over time
blocked_log = deque(maxlen=LOG_LEN)        # VACL / mitigation events
gateway_log = deque(maxlen=LOG_LEN)        # gateway access attempts


def _now():
    return time.time()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/telemetry", methods=["POST"])
def telemetry():
    """Called once per detection cycle by cnn_guard.py."""
    data = request.get_json(force=True, silent=True) or {}

    with LOCK:
        state["pkt_count"] = data.get("pkt_count", 0)
        state["fwd_pkts"] = data.get("fwd_pkts", 0)
        state["bwd_pkts"] = data.get("bwd_pkts", 0)
        state["fwd_len"] = data.get("fwd_len", 0)
        state["bwd_len"] = data.get("bwd_len", 0)
        state["prediction"] = float(data.get("prediction", 0.0))
        state["status"] = data.get("status", "SAFE")
        state["lockdown"] = bool(data.get("lockdown", False))
        state["attacker_ip"] = data.get("attacker_ip")
        state["last_update"] = _now()
        state["connected"] = True

        new_blocked = data.get("new_blocked_ip")
        if new_blocked:
            blocked_log.appendleft({"t": _now(), "ip": new_blocked})
            state["blocked_ips"] = data.get("blocked_ips", state["blocked_ips"])
        elif "blocked_ips" in data:
            state["blocked_ips"] = data["blocked_ips"]

        history.append({
            "t": _now(),
            "pkt_count": state["pkt_count"],
            "prediction": state["prediction"],
            "status": state["status"],
        })

    return jsonify({"ok": True})


@app.route("/api/gateway_event", methods=["POST"])
def gateway_event():
    """Called by gateway.py on every incoming service request."""
    data = request.get_json(force=True, silent=True) or {}
    with LOCK:
        gateway_log.appendleft({
            "t": _now(),
            "mode": data.get("mode", "normal"),
            "result": data.get("result", "unknown"),
            "detail": data.get("detail", ""),
            "user": data.get("user"),
        })
    return jsonify({"ok": True})


@app.route("/api/state", methods=["GET"])
def get_state():
    with LOCK:
        stale = state["last_update"] is not None and (_now() - state["last_update"]) > 5
        payload = dict(state)
        payload["connected"] = state["connected"] and not stale
        payload["history"] = list(history)
        payload["blocked_log"] = list(blocked_log)
        payload["gateway_log"] = list(gateway_log)
    return jsonify(payload)


if __name__ == "__main__":
    print("[*] SDN Guard Dashboard running at http://0.0.0.0:5050")
    app.run(host="0.0.0.0", port=5050, threaded=True)
