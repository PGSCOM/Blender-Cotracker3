# SPDX-License-Identifier: GPL-3.0-or-later

import math
import queue
import threading

import bpy
from bpy.types import Operator

from . import preferences
from .bridge import client, coords

GRID_PREFIX = "cotracker_grid_"

_JOB = {
    "running": False,
    "queue": None,
    "clip_name": None,
    "obj_name": None,
    "groups": None,  # [(track_name, point_count), ...]
    "f0": 0,  # clip frames of the tracked range
    "f1": 0,
    "clear_range": True,
}


def _ensure_service(url):
    """Return when the service answers /health; spawn it first if allowed."""
    try:
        client.health(url, timeout=5)
        preferences.SERVICE_STATUS.update(text="Service ready", ok=True)
        return
    except Exception:
        pass

    prefs = bpy.context.preferences.addons[__package__].preferences
    if not prefs.auto_start_service:
        raise client.ServiceError(
            "Inference service not reachable at %s. Start it from the "
            "add-on preferences or run: python service/server.py" % url
        )
    proc, log_path = preferences.spawn_service()
    preferences.SERVICE_STATUS["text"] = "Waiting for service (pid %d)..." % proc.pid
    client.wait_for_service(url)
    preferences.SERVICE_STATUS.update(text="Service ready", ok=True)


def _worker(url, payload):
    try:
        _ensure_service(url)
        data = client.track(url, payload)
        _JOB["queue"].put(("ok", data))
    except Exception as err:
        _JOB["queue"].put(("err", str(err)))


def _clear_range(track, f0, f1):
    for marker in list(track.markers):
        if f0 <= marker.frame <= f1:
            track.markers.delete_frame(marker.frame)


def _write_markers(data):
    clip = bpy.data.movieclips.get(_JOB["clip_name"])
    if clip is None:
        raise RuntimeError("clip %r no longer exists" % _JOB["clip_name"])
    tracking_obj = clip.tracking.objects.get(_JOB["obj_name"])
    if tracking_obj is None:
        raise RuntimeError("tracking object %r no longer exists" % _JOB["obj_name"])

    tracks_px, vis = client.decode_result(data)
    width, height = clip.size
    t0 = int(data["t0"])
    f_start = coords.video_to_blender_frame(
        t0, clip.frame_start, clip.frame_offset
    )
    written = 0
    offset = 0
    for name, count in _JOB["groups"]:
        track = tracking_obj.tracks.get(name)
        if track is None:
            offset += count
            continue
        if _JOB["clear_range"]:
            _clear_range(track, _JOB["f0"], _JOB["f1"])
        for j in range(vis.shape[0]):
            for k in range(count):
                idx = offset + k
                if not vis[j, idx]:
                    continue
                px, py = tracks_px[j, idx]
                if not (math.isfinite(px) and math.isfinite(py)):
                    continue
                bx, by = coords.px_to_blender(px, py, width, height)
                bx = min(max(bx, 0.0), 1.0)
                by = min(max(by, 0.0), 1.0)
                track.markers.insert_frame(f_start + j, co=(bx, by))
                written += 1
        offset += count
    print(
        "[CoTracker3] wrote %d markers across %d tracks"
        % (written, len(_JOB["groups"]))
    )


_STAGE_WEIGHT = {"reading": (0, 20), "tracking": (20, 100)}


def _poll_progress():
    """Ask the service how far along the current job is and update the UI."""
    try:
        info = client.progress(preferences.service_url(), timeout=0.5)
    except Exception:
        return
    stage, current, total = info.get("stage"), info.get("current", 0), info.get("total", 0)
    lo, hi = _STAGE_WEIGHT.get(stage, (0, 100))
    frac = current / total if total else 0.0
    pct = lo + (hi - lo) * frac
    bpy.context.window_manager.progress_update(pct)
    if stage == "reading":
        preferences.SERVICE_STATUS["text"] = "Reading frames %d/%d..." % (current, total)
    elif stage == "tracking":
        preferences.SERVICE_STATUS["text"] = "Tracking... chunk %d/%d" % (current, total)
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "CLIP_EDITOR":
                area.tag_redraw()


def _poll():
    wm = bpy.context.window_manager
    if not _JOB["running"]:
        return None
    try:
        kind, result = _JOB["queue"].get_nowait()
    except queue.Empty:
        _poll_progress()
        return 0.1
    preferences.SERVICE_STATUS["busy"] = False
    if kind == "ok":
        try:
            _write_markers(result)
            wm.progress_update(100)
            preferences.SERVICE_STATUS["text"] = "Done - markers written"
        except Exception as err:
            preferences.SERVICE_STATUS["text"] = "Write failed: %s" % err
    else:
        preferences.SERVICE_STATUS["text"] = "Error: %s" % result[:300]
    wm.progress_end()
    _JOB["running"] = False
    return None


def _start_tracking(context, clip, query_items):
    """query_items: [{"name", "frame", "nx", "ny"}, ...] (Blender frame + norm co)"""
    wm = context.window_manager
    settings = wm.cotracker3
    if _JOB["running"]:
        raise RuntimeError("A tracking job is already running")

    width, height = clip.size
    f0 = settings.frame_start if settings.frame_start > 0 else clip.frame_start
    f1 = (
        settings.frame_end
        if settings.frame_end > 0
        else clip.frame_start + clip.frame_duration - 1
    )
    i0 = coords.blender_frame_to_video(f0, clip.frame_start, clip.frame_offset)
    i1 = coords.blender_frame_to_video(f1, clip.frame_start, clip.frame_offset)
    if i1 < i0:
        raise RuntimeError("Invalid frame range: %d..%d" % (i0, i1))

    queries = []
    groups = []
    for item in query_items:
        index = coords.blender_frame_to_video(
            item["frame"], clip.frame_start, clip.frame_offset
        )
        index = min(max(index, i0), i1)
        x, y = coords.blender_to_px(item["nx"], item["ny"], width, height)
        queries.append([index, x, y])
        groups.append((item["name"], 1))

    payload = {
        "path": bpy.path.abspath(clip.filepath),
        "source": clip.source,
        "t0": i0,
        "t1": i1,
        "queries": queries,
        "mode": settings.mode.lower(),
        "direction": settings.direction,
    }

    _JOB.update(
        running=True,
        queue=queue.Queue(),
        clip_name=clip.name,
        obj_name=clip.tracking.objects.active.name,
        groups=groups,
        f0=f0,
        f1=f1,
        clear_range=settings.clear_range,
    )
    wm.progress_begin(0, 100)
    wm.progress_update(2)
    preferences.SERVICE_STATUS["busy"] = True
    preferences.SERVICE_STATUS["text"] = (
        "Tracking %d point(s), frames %d..%d (first run downloads the model)..."
        % (len(queries), f0, f1)
    )
    threading.Thread(
        target=_worker, args=(preferences.service_url(), payload), daemon=True
    ).start()
    bpy.app.timers.register(_poll, first_interval=0.25)
    return {"FINISHED"}


class CLIP_OT_cotracker3_track_selected(Operator):
    bl_idname = "clip.cotracker3_track_selected"
    bl_label = "Track Selected Markers with CoTracker3"
    bl_description = "Track all selected tracks with CoTracker3, starting at the current clip frame"

    def execute(self, context):
        space = getattr(context, "space_data", None)
        clip = getattr(space, "clip", None)
        if clip is None:
            self.report({"ERROR"}, "Open a movie clip in the Movie Clip Editor first")
            return {"CANCELLED"}
        clip_user = getattr(space, "clip_user", None)
        query_frame = getattr(clip_user, "frame", None)
        if query_frame is None:
            query_frame = context.scene.frame_current
        items = []
        for track in clip.tracking.objects.active.tracks:
            if not track.select:
                continue
            marker = track.markers.find_frame(query_frame, exact=True)
            if marker is not None:
                items.append(
                    {"name": track.name, "frame": query_frame,
                     "nx": marker.co[0], "ny": marker.co[1]}
                )
        if not items:
            self.report(
                {"ERROR"},
                "No selected tracks with a marker at frame %d. "
                "Place a marker at the current frame first." % query_frame,
            )
            return {"CANCELLED"}
        try:
            return _start_tracking(context, clip, items)
        except Exception as err:
            self.report({"ERROR"}, str(err))
            return {"CANCELLED"}


class CLIP_OT_cotracker3_track_grid(Operator):
    bl_idname = "clip.cotracker3_track_grid"
    bl_label = "Track New Grid with CoTracker3"
    bl_description = "Seed a dense grid of new tracks and track them with CoTracker3"

    def execute(self, context):
        space = getattr(context, "space_data", None)
        clip = getattr(space, "clip", None)
        if clip is None:
            self.report({"ERROR"}, "Open a movie clip in the Movie Clip Editor first")
            return {"CANCELLED"}
        if _JOB["running"]:
            self.report({"ERROR"}, "A tracking job is already running")
            return {"CANCELLED"}
        settings = context.window_manager.cotracker3
        tracking = clip.tracking.objects.active
        query_frame = (
            settings.grid_query_frame if settings.grid_query_frame > 0
            else clip.frame_start
        )

        for track in list(tracking.tracks):
            if track.name.startswith(GRID_PREFIX):
                tracking.tracks.remove(track)

        items = []
        for nx, ny in coords.grid_points(settings.grid_size):
            name = "%s%03d" % (GRID_PREFIX, len(items))
            track = tracking.tracks.new(name, frame=query_frame)
            marker = track.markers.find_frame(query_frame, exact=True)
            if marker is not None:
                marker.co = (nx, ny)
            items.append({"name": name, "frame": query_frame, "nx": nx, "ny": ny})
        try:
            return _start_tracking(context, clip, items)
        except Exception as err:
            self.report({"ERROR"}, str(err))
            return {"CANCELLED"}


classes = (CLIP_OT_cotracker3_track_selected, CLIP_OT_cotracker3_track_grid)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
