# SPDX-License-Identifier: GPL-3.0-or-later
# CoTracker3 inference for the Blender add-on. Runs OUTSIDE Blender,
# in a Python environment with torch installed.
#
# Video:  read as [1, T, 3, H, W] float32, values 0..255 (native frames).
# Queries: [N, 3] as float32 (t, x, y) with 0-based video frame index and
#          pixel coordinates (origin top-left), full resolution.
# Output: tracks [T, N, 2] float32 px, vis [T, N] bool.

from __future__ import annotations

import glob
import os
import re

import imageio.v3 as iio
import numpy as np
import torch

from direction_mask import apply_direction_mask, online_chunks

SUPPORTED_SOURCES = ("MOVIE", "SEQUENCE")

_models = {}
_device = None

PROGRESS = {"stage": "idle", "current": 0, "total": 0}


def device():
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def get_model(mode):
    """Load the cotracker3_offline / cotracker3_online model once."""
    if mode not in _models:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_%s" % mode)
        _models[mode] = model.to(device()).eval()
    return _models[mode]


def sequence_files(path):
    """Sorted files for an image sequence, from the first frame's path."""
    directory, base = os.path.split(path)
    pattern = re.sub(r"\d+", "*", base)
    return sorted(glob.glob(os.path.join(directory, pattern)))


def read_video(path, source, t0, t1):
    """Read video frames [t0..t1] inclusive as [1, T, 3, H, W] float32 0..255."""
    if source not in SUPPORTED_SOURCES:
        raise ValueError("Unsupported clip source %r" % source)

    total = t1 - t0 + 1
    PROGRESS.update(stage="reading", current=0, total=total)
    if source == "SEQUENCE":
        files = sequence_files(path)[t0 : t1 + 1]
        if not files:
            raise ValueError("No sequence files found for %r" % path)
        frames = []
        for i, f in enumerate(files):
            frames.append(np.asarray(iio.imread(f)))
            PROGRESS["current"] = i + 1
        frames = np.stack(frames)
    else:
        frames = []
        for index, frame in enumerate(iio.imiter(path, plugin="FFMPEG")):
            if index < t0:
                continue
            if index > t1:
                break
            frames.append(np.asarray(frame))
            PROGRESS["current"] = index - t0 + 1
        if not frames:
            raise ValueError("No frames found in %r" % path)
        frames = np.stack(frames)

    if frames.dtype != np.uint8:
        if frames.dtype.kind == "f":
            # Float images (EXR etc.) are typically normalized 0..1; scale by
            # their own max instead of assuming an 8/16-bit integer range.
            peak = float(frames.max())
            scale = 255.0 / peak if peak > 0 else 1.0
            frames = np.clip(frames.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        elif frames.max() > 255.0:
            frames = (frames.astype(np.float32) * (255.0 / 65535.0)).astype(np.uint8)
        else:
            frames = np.clip(frames, 0, 255).astype(np.uint8)
    video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float()
    return video.to(device())


def _run_offline(video, queries, backwards):
    model = get_model("offline")
    # The offline model runs as a single opaque pass; no sub-progress available.
    PROGRESS.update(stage="tracking", current=0, total=1)
    out_tracks, out_vis = model(video, queries=queries, backward_tracking=backwards)
    PROGRESS["current"] = 1
    return out_tracks[0], out_vis[0]


def _run_online(video, queries, flipped):
    """Streaming forward pass; window outputs are stitched so every frame gets
    the prediction from a window where it is not the initialization region."""
    model = get_model("online")
    step = model.step
    if flipped:
        video = video.flip(1)
        queries = queries.clone()
        queries[:, 0] = video.shape[1] - 1 - queries[:, 0]

    model(video_chunk=video, is_first_step=True, grid_size=0, queries=queries)
    chunks = online_chunks(video.shape[1], step)
    PROGRESS.update(stage="tracking", current=0, total=len(chunks))
    all_tracks, all_vis = [], []
    for i, (ind, end, keep_start, keep_len) in enumerate(chunks):
        chunk = video[:, ind:end]
        if chunk.shape[1] < 2 * step:
            pad = video[:, -1:].expand(-1, 2 * step - chunk.shape[1], -1, -1, -1)
            chunk = torch.cat([chunk, pad], dim=1)
        tracks, vis = model(video_chunk=chunk, grid_size=0, queries=queries)
        keep = slice(keep_start, keep_start + keep_len)
        all_tracks.append(tracks[0, keep])
        all_vis.append(vis[0, keep])
        PROGRESS["current"] = i + 1

    tracks = torch.cat(all_tracks, dim=0)
    vis = torch.cat(all_vis, dim=0)
    if flipped:
        tracks = tracks.flip(0)
        vis = vis.flip(0)
    return tracks, vis


def track(path, source, t0, t1, queries, mode, direction):
    """Run CoTracker3 over frames [t0..t1] and return (tracks, vis).

    tracks: np float32 [T, N, 2] pixel coords (origin top-left).
    vis:    np bool [T, N], True where the model reports a valid prediction.
    """
    t0, t1 = int(t0), int(t1)
    if t1 < t0:
        raise ValueError("frame range empty: t0=%d t1=%d" % (t0, t1))
    if mode == "online" and direction == "BOTH":
        raise ValueError("direction=BOTH is not supported in online mode (forward only)")
    video = read_video(path, source, t0, t1)
    T = video.shape[1]

    queries = torch.tensor(queries, dtype=torch.float32, device=device())[None]
    queries[:, :, 0].clamp_(0, T - 1)
    q_times = [int(t) for t in queries[0, :, 0].tolist()]

    backwards = direction in ("BACKWARD",) or (direction == "BOTH" and mode == "offline")
    if mode == "offline":
        tracks, vis = _run_offline(video, queries, backwards)
    else:
        tracks, vis = _run_online(video, queries, flipped=direction == "BACKWARD")

    vis = vis.bool()
    if direction == "BACKWARD" and mode == "online":
        # The online model only tracks forward; mask in flipped-video space.
        vis = apply_direction_mask(vis, [T - 1 - t for t in q_times], "FORWARD")
    else:
        vis = apply_direction_mask(vis, q_times, direction)

    PROGRESS.update(stage="idle", current=0, total=0)
    return (
        tracks.detach().cpu().numpy().astype(np.float32),
        vis.detach().cpu().numpy(),
    )