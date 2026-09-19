# SPDX-License-Identifier: GPL-3.0-or-later

import os
import tempfile

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import AddonPreferences, Operator, PropertyGroup

from .bridge import client

SERVICE_STATUS = {"ok": False, "text": "Service not checked", "busy": False}


def service_url():
    return bpy.context.preferences.addons[__package__].preferences.service_url.rstrip("/")


def server_log_path():
    return os.path.join(
        tempfile.gettempdir(),
        "cotracker3_service_%d.log" % os.getpid(),
    )


def spawn_service():
    prefs = bpy.context.preferences.addons[__package__].preferences
    server_py = os.path.join(os.path.dirname(__file__), "service", "server.py")
    if not os.path.isfile(server_py):
        raise FileNotFoundError("service/server.py not found next to the add-on")
    log_path = server_log_path()
    proc = client.spawn_service(
        server_py, prefs.python_executable, prefs.service_url, log_path
    )
    return proc, log_path


class CotrackerSettings(PropertyGroup):
    """Per-session tracking settings, stored on the WindowManager."""

    def _mode_update(self, context):
        if self.mode == "ONLINE" and self.direction == "BOTH":
            self.direction = "FORWARD"

    mode: EnumProperty(
        name="Mode",
        description="Offline: whole clip at once, tracks both directions. "
        "Online: streaming, forward only, less VRAM",
        items=[
            ("OFFLINE", "Offline", "Whole clip in one pass, both directions possible"),
            ("ONLINE", "Online", "Streaming forward pass, memory efficient"),
        ],
        default="OFFLINE",
        update=_mode_update,
    )
    direction: EnumProperty(
        name="Direction",
        items=[
            ("FORWARD", "Forward", "Track from the query frame forward"),
            ("BACKWARD", "Backward", "Track from the query frame backward"),
            ("BOTH", "Both", "Track both directions (offline mode only)"),
        ],
        default="BOTH",
    )
    grid_size: IntProperty(
        name="Grid Size", description="N x N points for the new grid", min=1, max=50, default=10
    )
    grid_query_frame: IntProperty(
        name="Grid Query Frame",
        description="Clip frame where grid points are seeded; 0 uses the clip start",
        default=0,
    )
    frame_start: IntProperty(
        name="Start", description="First clip frame to track; 0 uses the clip start", default=0
    )
    frame_end: IntProperty(
        name="End", description="Last clip frame to track; 0 uses the clip end", default=0
    )
    clear_range: BoolProperty(
        name="Replace Existing Markers",
        description="Delete existing markers inside the tracked range before writing results",
        default=True,
    )


class CotrackerPrefs(AddonPreferences):
    bl_idname = __package__

    service_url: StringProperty(
        name="Service URL",
        description="URL of the local CoTracker3 inference service",
        default="http://127.0.0.1:8700",
    )
    python_executable: StringProperty(
        name="Python Executable",
        description="Python (with torch installed) used to launch the service; "
        "leave empty to use 'python' from PATH",
        subtype="FILE_PATH",
    )
    auto_start_service: BoolProperty(
        name="Auto-start Service",
        description="Launch the service automatically when tracking and it is not running",
        default=True,
    )

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        col = layout.column()
        col.prop(self, "service_url")
        col.prop(self, "python_executable")
        col.prop(self, "auto_start_service")
        col.separator()
        row = layout.row()
        row.operator("cotracker3.start_service", text="Start Service")
        row.operator("cotracker3.check_service", text="Check Connection")
        layout.label(text="Service status: " + SERVICE_STATUS["text"])


class CLIP_OT_cotracker3_start_service(Operator):
    bl_idname = "cotracker3.start_service"
    bl_label = "Start CoTracker3 Service"
    bl_description = "Launch the local inference service in the background"

    def execute(self, context):
        try:
            proc, log_path = spawn_service()
            SERVICE_STATUS["ok"] = False
            SERVICE_STATUS["text"] = "Starting (pid %d)... log: %s" % (proc.pid, log_path)
        except Exception as err:
            SERVICE_STATUS["text"] = "Start failed: %s" % err
            self.report({"ERROR"}, str(err))
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            "Service starting (pid %d). First track request downloads the model."
            % proc.pid,
        )
        return {"FINISHED"}


class CLIP_OT_cotracker3_check_service(Operator):
    bl_idname = "cotracker3.check_service"
    bl_label = "Check CoTracker3 Service"

    def execute(self, context):
        try:
            info = client.health(service_url(), timeout=5)
            SERVICE_STATUS["ok"] = True
            SERVICE_STATUS["text"] = (
                "Connected - device: %s, models: %s"
                % (info.get("device", "?"), ", ".join(info.get("models", [])) or "none")
            )
            self.report({"INFO"}, SERVICE_STATUS["text"])
        except Exception as err:
            SERVICE_STATUS["ok"] = False
            SERVICE_STATUS["text"] = "Not reachable: %s" % err
            self.report({"WARNING"}, SERVICE_STATUS["text"])
        if context.area:
            context.area.tag_redraw()
        return {"FINISHED"}


classes = (CotrackerSettings, CotrackerPrefs, CLIP_OT_cotracker3_start_service,
           CLIP_OT_cotracker3_check_service)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.WindowManager.cotracker3 = PointerProperty(type=CotrackerSettings)


def unregister():
    del bpy.types.WindowManager.cotracker3
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)