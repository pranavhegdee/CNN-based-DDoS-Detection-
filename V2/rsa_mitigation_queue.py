"""
rsa_mitigation_queue.py
-------------------------
gateway.py runs inside a Mininet HOST's network namespace (e.g. h1 at
10.0.0.1). The Ryu controller's REST API (127.0.0.1:8080) lives in a
DIFFERENT namespace -- typically the root namespace, alongside
cnn_guard1.py, which is why cnn_guard1.py's calls to Ryu have always
worked and gateway.py's never could. "127.0.0.1" inside h1 is h1's own
loopback, not the controller's.

Mininet hosts are network namespaces, not full containers, so they DO
share one filesystem. That makes a simple append-only file queue the
easiest reliable channel across the namespace boundary: gateway.py can
always write to it (same machine, no networking involved), and
cnn_guard1.py -- which already has a working path to Ryu -- polls it and
does the actual mitigation call itself, reusing the exact same code path
it already uses for network-layer attacks.

Format: one JSON object per line (JSONL), append-only.
"""

import json
import os
import time

QUEUE_PATH = "/tmp/rsa_mitigation_queue.jsonl"


def enqueue(ip, count, reason):
    """Called from gateway.py. Fire-and-forget; a failed write here just
    means this one escalation gets missed, not a crash."""
    entry = {"ip": ip, "count": count, "reason": reason, "ts": time.time()}
    try:
        with open(QUEUE_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[-] Failed to enqueue RSA mitigation request: {e}")


class QueueReader:
    """Called from cnn_guard1.py. Tracks how many lines have already been
    processed (in-memory; fine for a single long-running process / one
    Mininet demo run) so repeated polls don't reprocess old entries."""

    def __init__(self, path=QUEUE_PATH):
        self.path = path
        self._lines_read = 0

    def poll_new_entries(self):
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path) as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[-] Failed to read RSA mitigation queue: {e}")
            return []

        new_lines = lines[self._lines_read:]
        self._lines_read = len(lines)

        entries = []
        for line in new_lines:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries
