# SPDX-License-Identifier: GPL-3.0-or-later
# Blender <-> CoTracker3 coordinate conversions.
# Pure functions, no bpy imports (runs standalone for self-checks).

# Blender marker co: normalized 0..1, origin bottom-left.
# CoTracker3 tracks: pixel coordinates, origin top-left.


def blender_to_px(nx, ny, width, height):
    """Normalized Blender co (origin bottom-left) to pixel co (origin top-left)."""
    return nx * width, (1.0 - ny) * height


def px_to_blender(px, py, width, height):
    """Pixel co (origin top-left) to normalized Blender co (origin bottom-left)."""
    return px / width, 1.0 - py / height


def grid_points(grid_size):
    """Uniform grid of normalized points covering the frame, row-major."""
    return [
        ((gx + 0.5) / grid_size, (gy + 0.5) / grid_size)
        for gy in range(grid_size)
        for gx in range(grid_size)
    ]


def blender_frame_to_video(frame, frame_start, frame_offset):
    """Blender clip frame to 0-based video frame index.

    Blender maps clip frames onto the source footage with `frame_start`
    (first clip frame) and `frame_offset` (frames skipped at the start of
    the source footage). Default (start=1, offset=0) maps frame 1 to
    video index 0 for both movies and image sequences.
    """
    return frame - frame_start + frame_offset


def video_to_blender_frame(index, frame_start, frame_offset):
    """Inverse of blender_frame_to_video."""
    return index + frame_start - frame_offset


if __name__ == "__main__":
    # Self-check: coordinate round-trips against known values.
    W, H = 1920, 1080

    assert blender_to_px(0.0, 1.0, W, H) == (0.0, 0.0)  # top corner -> top-left px
    assert blender_to_px(1.0, 0.0, W, H) == (W, H)  # bottom corner -> bottom px
    assert blender_to_px(0.5, 0.5, W, H) == (960.0, 540.0)  # center stays center
    assert px_to_blender(0.0, 0.0, W, H) == (0.0, 1.0)  # px top-left -> norm top

    nx, ny = 0.25, 0.75
    px, py = blender_to_px(nx, ny, W, H)
    assert px_to_blender(px, py, W, H) == (nx, ny), "round-trip failed"

    pts = grid_points(2)
    assert len(pts) == 4
    assert all(0.0 <= a <= 1.0 and 0.0 <= b <= 1.0 for a, b in pts)
    assert pts[0] == (0.25, 0.25) and pts[3] == (0.75, 0.75)
    assert len(grid_points(1)) == 1

    # Frame mapping: default start=1, offset=0; and with offsets.
    assert blender_frame_to_video(1, 1, 0) == 0
    assert blender_frame_to_video(50, 1, 0) == 49
    assert video_to_blender_frame(49, 1, 0) == 50
    assert blender_frame_to_video(10, 5, 2) == 7
    assert video_to_blender_frame(7, 5, 2) == 10

    print("coords.py self-check OK")