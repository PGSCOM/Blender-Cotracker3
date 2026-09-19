#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Local HTTP service that runs CoTracker3 for the Blender add-on.
#
#   python server.py [--port 8700]
#
# Endpoints:
#   GET  /health -> {"status": "ok", "device": ..., "models": [...]}
#   POST /track  -> JSON payload, JSON + base64 response (see cotracker_runner.track)

from __future__ import annotations

import argparse
import base64
import json
import threading

import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cotracker_runner

INFER_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    timeout = 7200

    def log_message(self, *args):
        pass  # quiet; errors surface in the JSON response

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/progress":
            self._send(dict(cotracker_runner.PROGRESS))
            return
        if self.path != "/health":
            self._send({"error": "not found"}, 404)
            return
        self._send(
            {
                "status": "ok",
                "device": cotracker_runner.device(),
                "models": sorted(cotracker_runner._models),
                "torch_hub_download": None,
            }
        )

    def do_POST(self):
        if self.path != "/track":
            self._send({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as err:
            self._send({"error": "bad request: %s" % err}, 400)
            return

        try:
            with INFER_LOCK:
                tracks, vis = cotracker_runner.track(**payload)
            tracks = np.ascontiguousarray(tracks)
            vis = np.ascontiguousarray(vis).astype(bool)
            self._send(
                {
                    "ok": True,
                    "t0": int(payload["t0"]),
                    "t1": int(payload["t1"]),
                    "shape": [int(s) for s in tracks.shape],
                    "tracks_b64": base64.b64encode(tracks.tobytes()).decode("ascii"),
                    "vis_b64": base64.b64encode(vis.tobytes()).decode("ascii"),
                }
            )
        except Exception as err:
            cotracker_runner.PROGRESS.update(stage="idle", current=0, total=0)
            self._send({"ok": False, "error": "tracking failed: %s" % err}, 500)


def main():
    parser = argparse.ArgumentParser(description="CoTracker3 inference service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8700)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("CoTracker3 service listening on http://%s:%d" % (args.host, args.port))
    print("Device: %s" % cotracker_runner.device())
    server.serve_forever()


if __name__ == "__main__":
    main()