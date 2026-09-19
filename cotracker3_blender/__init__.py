# SPDX-License-Identifier: GPL-3.0-or-later
# Metadata (name, version, description) lives in blender_manifest.toml -
# this is a Blender 4.2+ Extension, bl_info is deprecated/redundant here.

from . import operators, panels, preferences

modules = (preferences, operators, panels)


def register():
    for module in modules:
        module.register()


def unregister():
    for module in reversed(modules):
        module.unregister()


if __name__ == "__main__":
    register()