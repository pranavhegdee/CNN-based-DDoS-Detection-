# SDN Guard Dashboard

A live, light-themed web dashboard for your CNN traffic-guard + Ryu mitigation
pipeline. It doesn't change your detection logic or your mitigation logic —
`cnn_guard.py` and `gateway.py` just send small JSON pings to a new
`dashboard_server.py`, which keeps a short rolling history in memory and
serves the UI.

## Files

| File                  | Role                                                              |
|-----------------------|--------------------------------------------------------------------|
| `dashboard_server.py` | New. Flask app that aggregates state and serves the UI (port 5050) |
| `templates/index.html`| New. The dashboard itself                                          |
| `cnn_guard.py`        | Your file + ~15 lines to POST telemetry after each detection cycle |
| `gateway.py`          | Your file + a few lines to POST each access attempt (async, non-blocking) |
| `ryu_mit.py`          | **Unchanged** — no edits needed                                    |

## What you get

- **Live topology panel** — host → switch (s1) → server (10.0.0.1), animated
  packet flow, the switch pulses red the moment an attack is flagged, and a
  lock badge appears on the server node while the gateway is in RSA lockdown.
- **Metric cards** — packets/sec, fwd/bwd packet counts, model confidence
  (with a colored gauge), and number of IPs currently dropped at the switch.
- **Traffic + anomaly chart** — rolling ~2-minute view of packet rate vs. the
  CNN's confidence score.
- **VACL mitigation log** — every IP the Ryu controller has dropped.
- **Gateway access log** — every request the Flask gateway handled, whether
  it was the open channel or a signed RSA check, success or denial.

## Running it

Start the dashboard first (or anytime — the other scripts just retry
silently if it isn't up yet):

```bash
pip install flask
python3 dashboard_server.py
```

Open **http://127.0.0.1:5050** in a browser (do this on the machine where
you can reach that port — e.g. your Mininet host's desktop, or tunnel the
port if you're on a headless VM).

Then run the rest of your stack exactly as before:

```bash
ryu-manager ryu_mit.py
python3 gateway.py
sudo python3 cnn_guard.py
```

Everything shows up on the dashboard within ~1 second.

## Notes

- If `cnn_guard.py`/`gateway.py` run on a different host/namespace than the
  dashboard, set `DASHBOARD_URL` before launching them, e.g.:
  `DASHBOARD_URL=http://192.168.1.10:5050 python3 cnn_guard.py`
- Set `DASHBOARD_URL=""` on either script to fully disable reporting without
  touching any other code.
- Reporting is wrapped in `try/except` with short timeouts everywhere, so if
  the dashboard is down or unreachable, detection and mitigation keep running
  exactly as before — the UI just goes quiet.
