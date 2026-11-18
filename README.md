# Blender CoTracker3 Tracking

[Demo](https://youtu.be/j6hVbNbe9MM)

Add-on for Blender 4.2+ that replaces the Movie Clip Editor's internal tracker
(KLT) with [CoTracker3](https://github.com/facebookresearch/co-tracker), Meta
AI's transformer point tracker. Blender keeps its native track/marker data
structures, so camera solve, stabilization, etc. keep working unchanged -
only the "compute track positions" step is delegated to CoTracker3, running
in a separate Python process with PyTorch.

## Architecture

```
cotracker3_blender/            <- the Blender extension
├── blender_manifest.toml
├── __init__.py  preferences.py  operators.py  panels.py
├── bridge/
│   ├── coords.py              Blender <-> pixel coordinate conversion
│   └── client.py              HTTP client (stdlib + numpy, runs inside Blender)
└── service/                   <- NOT executed inside Blender
    ├── server.py              local HTTP service (stdlib http.server)
    ├── cotracker_runner.py    video decoding + CoTracker3 offline/online
    ├── direction_mask.py      forward/backward masking logic
    ├── requirements.txt
    └── smoke_test.py          protocol self-test (no torch needed)
```

The add-on talks to `http://127.0.0.1:8700` over loopback. The service decodes
the clip footage itself (imageio/FFmpeg) and returns tracked positions, which
the add-on writes back as native `MovieTrackingMarker`s.

## Setup

1. Blender side: install the extension (zip or directory) via
   Preferences -> Extensions.

2. Service side (one-time, separate Python environment - NOT Blender's Python):

   ```
   python -m venv .venv
   .venv\Scripts\activate            # Windows  (linux: .venv/bin/activate)
   pip install -r cotracker3_blender/service/requirements.txt
   ```

   Requires Python 3.10+ with a matching torch wheel. For CUDA install
   torch first from https://pytorch.org/get-started/locally/ , then the rest.

3. Start the service (same venv):

   ```
   python cotracker3_blender/service/server.py --port 8700
   ```

   Or let the add-on start it: preferences -> add-on -> set "Python
   Executable" to the venv's python and enable "Auto-start Service".

The first tracking request downloads the model weights (~1.3 GB for both
models) from Hugging Face into the torch hub cache. Run
Extensions -> "CoTracker3 Tracking" -> "Check Connection" to show the device
and loaded models.

## Usage

In the Movie Clip Editor (Tracking mode, Clip view):

1. Place markers as usual (or click "Track New Grid" to seed a dense grid).
2. Sidebar -> Track -> CoTracker3:
   - Mode: Offline (whole clip at once, tracks both directions, better
     quality) or Online (streaming, forward only, less VRAM - recommended
     for long clips).
   - Direction: Forward / Backward / Both (Both requires offline mode).
   - Start/End: clip frame range to process (0 = full clip).
3. Click "Track Selected Markers" (tracks the selected tracks, starting at
   the current clip frame) or "Track New Grid".

Results land in the native markers: frames where CoTracker3 reports the point
as occluded simply get no marker. You can then run `Solve Camera Motion` as
usual.

## Notes and limitations

- **Frames**, not time: CoTracker3 works on the clip's native frames.
  Blender's `frame_start` / `frame_offset` are honored when mapping clip
  frames to footage frames; FPS settings don't change the mapping.
- **Image sequences**: supported (frames are read as files); the model
  needs 8-bit data, so 16-bit/EXR/raw sequences are converted or skipped -
  convert to 8-bit PNG/TIFF if tracking fails on them.
- **Undo**: marker writes are not on the undo stack. "Replace Existing
  Markers" only rewrites the processed range, and grid runs only replace
  `cotracker_grid_*` tracks.
- Tracked range is processed in one request; very long clips in Offline mode
  need a lot of VRAM - prefer Online mode for those.
- The tracking job runs in a background thread (Blender never blocks, use
  the progress bar), but only one job can run at a time.

## Developing

Self-checks (need only numpy):

```
python cotracker3_blender/bridge/coords.py
python cotracker3_blender/service/direction_mask.py
python cotracker3_blender/service/smoke_test.py
```

Package the extension:

```
blender --command extension build cotracker3_blender
```
