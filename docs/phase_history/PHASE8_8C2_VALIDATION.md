# Phase 8.8C2 — Live Camera Trajectory Visualization

## Scope

Phase 8.8C2 promotes the trajectory buffer introduced in C1 into a visible live camera path without changing reconstruction authority or backend execution. It also improves orientation presentation in two distinct layers:

- **world-origin axes remain in the 3D scene** and now use restrained, path-adaptive lengths rather than the visually dominant generic `XYZAxis`;
- a **fixed bottom-right orientation gizmo** shows viewer X/Y/Z orientation in screen space while the bottom-left `VIEWER Z-UP • ARBITRARY GAUGE` safety badge remains unchanged.

The trajectory is presentation-only. It uses the already coalesced UI pose stream, retains at most 20,000 UI samples, and decimates only the rendered line to at most 4,096 points. No pose, scale, calibration, metrology, or backend contract is modified.

## Production changes

- `app/reconstruction_visualization.py`
  - orientation-gizmo projection helper;
  - restrained adaptive world-axis sizing;
  - deterministic trajectory display decimation.
- `app/widgets/viewport/vispy_canvas.py`
  - live trajectory visible by default;
  - start marker and bounded/decimated line rendering;
  - restrained custom world-origin axes;
  - orientation-gizmo projection access.
- `app/widgets/reconstruction_orientation_gizmo.py`
  - fixed bottom-right read-only X/Y/Z viewer-orientation indicator.
- `app/widgets/reconstruction_page.py`
  - trajectory visibility toolbar toggle;
  - orientation gizmo placement;
  - C2 viewport guidance text;
  - explicit gizmo timer shutdown before canvas close.
- `app/icons.py`
  - trajectory toolbar icon.

No files under `core/reconstruction/` are changed. No Stella, PyCOLMAP, DA3, Gaussian Splatting, or NeRF backend code is changed.

## Safety / authority invariants

- pose convention remains `T_reconstruction_from_camera`;
- viewer transform remains `X=reconstruction X, Y=reconstruction Z, Z=-reconstruction Y`;
- scale remains arbitrary unless established by existing evidence-driven authority;
- Test Mode remains non-metric and metrology-not-certified;
- entering Reconstruction still does not start a backend;
- trajectory visibility can be toggled without affecting capture or reconstruction runtime;
- trajectory history is cleared for a new live session but the user's visibility preference is preserved;
- stopped live results retain the final camera pose and trajectory until a new session starts.

## Automated validation

Focused C2 gate:

```bash
./scripts/run_phase8_8c2_tests.sh quick
```

Expected: 29 tests pass (C1 foundation migration + C2 unit/UI coverage).

Backward gate:

```bash
./scripts/run_phase8_8c2_tests.sh backward
```

Expected B5/B4 focused gates remain green.

## Manual preview — no backend

```bash
python scripts/run_phase8_8c2_trajectory_preview.py
```

Expected visual checks:

- cyan live camera trajectory is visible;
- red current-camera frustum remains visible;
- restrained world-origin X/Y/Z axes remain at the true scene origin;
- bottom-right orientation gizmo is visible and follows orbit/view changes;
- bottom-left `VIEWER Z-UP • ARBITRARY GAUGE` badge remains visible;
- trajectory toolbar button hides/shows only the path;
- Orbit/Pan/Zoom/Fit/Focus/Reset/Grid controls remain functional.

## Real Stella → VisPy acceptance

Run from a normal desktop session (not `QT_QPA_PLATFORM=offscreen`):

```bash
rm -rf /tmp/phase8_8c2_live_trajectory
python scripts/run_phase8_8c2_live_trajectory_smoke.py \
  --max-frames 500 \
  --playback-speed 2.0
```

Required:

```text
Phase 8.8C2 live trajectory smoke: PASS
```

The summary must show visible trajectory rendering, more than one retained/displayed path sample, a finite current camera position, restrained world-axis length, orientation-gizmo readiness, clean stop/cleanup, arbitrary scale, uncertified metrology in Test Mode, and no controller errors.

## Commit gate

Do not commit until:

1. `git diff --check` is clean;
2. focused and backward tests pass;
3. no-backend preview is visually accepted;
4. real Stella live trajectory smoke passes;
5. normal `python main.py` launch/switch/close sanity passes.
