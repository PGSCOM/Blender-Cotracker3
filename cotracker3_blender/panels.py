# SPDX-License-Identifier: GPL-3.0-or-later

import bpy
from bpy.types import Panel

from . import preferences


class CLIP_PT_cotracker3(Panel):
    bl_space_type = "CLIP_EDITOR"
    bl_region_type = "UI"
    bl_category = "Track"
    bl_label = "CoTracker3"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return space and space.clip and space.mode == "TRACKING" and space.view == "CLIP"

    def draw(self, context):
        layout = self.layout
        settings = context.window_manager.cotracker3

        status = preferences.SERVICE_STATUS
        icon = "CHECKMARK" if status["ok"] else ("TIME" if status["busy"] else "ERROR")
        layout.label(text="Service: " + status["text"], icon=icon)

        box = layout.box()
        box.label(text="Settings", icon="SETTINGS")
        box.prop(settings, "mode")
        row = box.row()
        row.prop(settings, "direction")
        row.enabled = settings.mode == "OFFLINE" or settings.direction != "BOTH"
        col = box.column(align=True)
        col.prop(settings, "frame_start")
        col.prop(settings, "frame_end")
        box.prop(settings, "clear_range")

        grid = layout.box()
        grid.label(text="New Grid", icon="GRID")
        grid.prop(settings, "grid_size")
        grid.prop(settings, "grid_query_frame")

        col = layout.column(align=True)
        col.scale_y = 1.6
        col.operator("clip.cotracker3_track_selected", icon="TRACKING_FORWARDS")
        col.operator("clip.cotracker3_track_grid", icon="GRID")

        layout.operator("cotracker3.check_service", text="Check Service", icon="FILE_REFRESH")


classes = (CLIP_PT_cotracker3,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)