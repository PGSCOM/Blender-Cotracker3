#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Protocol smoke test: verifies the HTTP + base64 contract between Blender
# (bridge/client.py) and the service (server.py) WITHOUT torch. Run with the
# service's own Python:  python smoke_test.py

from __future__ import annotations

import os
import socket
import sys
import threading
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Fake runner: same interface as cotracker_runner, no torch.
fake_runner = types.ModuleType("cotracker_runner")


def _fake_track(path, source, t0, t1, queries, mode, direction):
    n = len(queries)
    T = t1 - t0 + 1
    tracks = np.zeros((T, n, 2), dtype=np.float32)
    vis = np.ones((T, n), dtype=bool)
    for k, (t, x, y) in enumerate(queries):
        t = int(t) - t0  # relative to window, like the real runner
        tracks[:, k, 0] = x + np.arange(T)
        tracks[:, k, 1] = y
        if direction == "FORWARD":
            vis[:t, k] = False
        elif direction == "BACKWARD":
            vis[t + 1 :, k] = False
    return tracks, vis


fake_runner.track = _fake_track
fake_runner.device = lambda: "fake"
fake_runner._models = {}
sys.modules["cotracker_runner"] = fake_runner

import server  # noqa: E402  (imports the fake cotracker_runner)

sys.path.insert(0, os.path.join(HERE, "..", "bridge"))
import client  # noqa: E402  (Blender-side client, stdlib + numpy only)

# Free port.
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
port = sock.getsockname()[1]
sock.close()

httpd = server.ThreadingHTTPServer(("127.0.0.1", port), server.Handler)
thread = threading.Thread(target=httpd.serve_forever, daemon=True)
thread.start()

url = "http://127.0.0.1:%d" % port

health = client.health(url)
assert health["status"] == "ok" and health["device"] == "fake", health

payload = {
    "path": "fake.mp4",
    "source": "MOVIE",
    "t0": 10,
    "t1": 15,
    "mode": "offline",
    "direction": "FORWARD",
    "queries": [[10, 100.0, 50.0], [12, 200.0, 80.0]],  # (t, x, y) video idx
}
data = client.track(url, payload)
tracks, vis = client.decode_result(data)

T, N, _ = tracks.shape
assert T == 6 and N == 2, (T, N)
assert data["t0"] == 10 and data["t1"] == 15
assert vis[:, 0].all()  # query at t=10 == window start, nothing before it
assert not vis[0:2, 1].any() and vis[2:, 1].all()  # query at t=12 (local 2)
assert tracks[3, 0, 0] == 100.0 + 3  # diagonal drift recovered

bad = None
try:
    client.track(url, {"queries": "not-a-list"})
    raise AssertionError("expected ServiceError for bad payload")
except client.ServiceError:
    pass

httpd.shutdown()
print("smoke_test.py OK")