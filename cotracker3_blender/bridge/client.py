# SPDX-License-Identifier: GPL-3.0-or-later
# HTTP client for the local CoTracker3 inference service.
# Uses only stdlib + numpy (bundled with Blender).

from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np

TRACK_TIMEOUT = 7200  # first run downloads the model, be generous


class ServiceError(RuntimeError):
    pass


def health(url, timeout=5.0):
    with urllib.request.urlopen(url + "/health", timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def progress(url, timeout=2.0):
    with urllib.request.urlopen(url + "/progress", timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def track(url, payload, timeout=TRACK_TIMEOUT):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url + "/track",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        try:
            detail = json.loads(err.read().decode("utf-8")).get("error", str(err))
        except Exception:
            detail = str(err)
        raise ServiceError(detail) from err


def decode_result(data):
    """Decode a /track response into (tracks [T,N,2] float32 px, vis [T,N] bool)."""
    tracks = np.frombuffer(
        base64.b64decode(data["tracks_b64"]), dtype=np.float32
    ).reshape(data["shape"])
    vis = np.frombuffer(
        base64.b64decode(data["vis_b64"]), dtype=np.bool_
    ).reshape(data["shape"][0], data["shape"][1])
    return tracks, vis


def spawn_service(server_py, python_exe, url, log_path):
    """Launch the inference service as a detached background process."""
    import shutil

    py = python_exe if python_exe.strip() else (shutil.which("python") or "")
    if not py:
        raise ServiceError(
            "Python executable not found. Set it in the add-on preferences."
        )
    host_port = url.rsplit("/", 1)[-1]
    port = host_port.rsplit(":", 1)[-1] if ":" in host_port else "8700"
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    log = open(log_path, "a", buffering=1)
    return subprocess.Popen(
        [py, server_py, "--port", port],
        stdout=log,
        stderr=log,
        stdin=subprocess.DEVNULL,
        **kwargs,
    )


def wait_for_service(url, timeout=300.0, interval=2.0):
    """Block until /health answers or timeout. Raises ServiceError on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            health(url, timeout=1000)
            return
        except Exception:
            time.sleep(interval)
    raise ServiceError(
        "Inference service did not start within %.0fs. "
        "Check the Python executable and that torch/imageio are installed." % timeout
    )