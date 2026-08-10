# Phase 8.8B2 — Reconstruction Source & Workflow Controls

## Scope

B2 activates source/workflow selection on the dormant Reconstruction page while keeping runtime actions and primary Reconstruction navigation disabled.

Supported selections:

- **Live source** — reuses the existing `ApplicationController` Camera / RTSP / Video capture object. It never creates a second capture path.
- **Image sequence** — selects one top-level directory and requires at least three supported image files.
- **Recorded video** — selects one existing supported video file.

Selection is represented by `ReconstructionSourceSelection` in the existing `ReconstructionController`. Selection is dependency-light and starts no Stella, PyCOLMAP, DA3, Gaussian, or NeRF backend.

## Safety invariants

- `AppMode.RECONSTRUCTION` remains outside `_ENABLED_MODES`.
- No Reconstruction routing branch is added to `MainWindow._on_mode_changed()`.
- Start / Pause / Stop remain disabled until B3.
- Live workflow borrows only the existing Live 3D source identity/status; capture ownership stays with `ApplicationController`.
- Static images are rejected as live-SLAM inputs and directed to the image-sequence workflow.
- Image-directory and video selections are validated before becoming controller state.
- Source selection cannot change while Reconstruction is active.
- Cancelling a file/folder dialog preserves the previous valid selection.
- No `core/reconstruction` schema or algorithm changes.
- No heavy backend imports are introduced into production `depth`.

## Required validation

```bash
./scripts/run_phase8_8b2_tests.sh quick
./scripts/run_phase8_8b2_tests.sh backward
python scripts/run_phase8_8b1_reconstruction_page_preview.py
python main.py
```

Manual preview checks:

1. Reconstruction remains disabled in normal navigation.
2. Direct B1 preview shows the three source buttons enabled.
3. Selecting **Live source** with no connected Live 3D source reports `Needs attention` and starts nothing.
4. Selecting a valid image directory reports its supported image count and `Ready`.
5. Selecting a valid recorded video reports the file name and `Ready`.
6. Start / Pause / Stop remain disabled.
7. No reconstruction backend starts from source selection alone.
