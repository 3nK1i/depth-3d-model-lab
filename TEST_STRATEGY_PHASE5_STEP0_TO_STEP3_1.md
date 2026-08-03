# Depth & 3D Model Lab — Test Strategy

Scope: Phase 5 Step 0 through Step 3.1.

The test pack is designed for the current architecture: PyQt6 UI, a spawned
inference worker, Development Mock, MoGe-2, VisPy point rendering, hardware
profiles, latest-frame backpressure, static-image lifecycle, telemetry, and
viewport controls.

## Principles

1. Automated tests never download model weights.
2. Automated tests never require a physical camera or RTSP endpoint.
3. The real worker integration test uses only Development Mock on CPU.
4. QSettings is redirected to a temporary directory so tests cannot overwrite
   real application preferences.
5. OpenGL visuals are validated manually; structural UI tests run offscreen.
6. Real RTX 2060/RTX 5090, CUDA, RTSP, and visual geometry checks remain manual.

## Test layers

### Smoke

Fast contract checks after every file replacement:

```bash
./scripts/run_phase5_tests.sh smoke
```

### Unit

Pure logic and isolated adapter/capture tests:

```bash
./scripts/run_phase5_tests.sh unit
```

### UI

Qt state, layout, controls, footer, source views, and settings:

```bash
./scripts/run_phase5_tests.sh ui
```

The expected OpenGL warnings under the offscreen platform are harmless.

### Integration

Starts the real spawned inference process, loads Development Mock, submits a
frame, receives depth, stops, restarts, and shuts down:

```bash
./scripts/run_phase5_tests.sh integration
```

### Full regression

```bash
./scripts/run_phase5_tests.sh full
```

### Coverage

Install `requirements/test.txt`, then run:

```bash
./scripts/run_phase5_tests.sh coverage
```

Open `htmlcov/index.html` locally.

## Step-by-step coverage map

### Step 0 — Runtime stabilization

- `tests/ui/test_state_machine.py`
- `tests/unit/test_session_tracker.py`
- `tests/ui/test_command_watchdog.py`
- `tests/ui/test_process_manager_unit.py`
- `tests/unit/test_worker_helpers.py`
- `tests/integration/test_inference_lifecycle.py`
- `tests/ui/test_main_window_shell.py`
- `tests/ui/test_widgets.py`

Validates legal transitions, centralized button states, stale-session rejection,
latest queue replacement, Stop acknowledgement, IPC disposal, spawn lifecycle,
restart, and window shell behavior.

### Step 1 — Hardware detection and profiles

- `tests/unit/test_gpu_info.py`
- `tests/unit/test_gpu_profiles.py`
- `tests/ui/test_hardware_profile_ui.py`
- `tests/ui/test_settings_persistence.py`

Validates CPU fallback, NVIDIA parsing, RTX 2060 Efficient, RTX 5090 Ultra,
unsafe-profile downgrade, precision support, persistent model/profile choices,
and the MODEL card.

### Step 1.1 — Active model and truthful metrics

- `tests/ui/test_main_window_shell.py`
- `tests/unit/test_performance_formatting.py`
- `tests/ui/test_performance_bar.py`

Validates Development Mock versus resolved MoGe model labels, real point count,
precision, and placeholder behavior.

### Step 1.2 — Live-stream performance guard

- `tests/unit/test_live_sampling.py`
- `tests/unit/test_geometry_sampling.py`
- `tests/ui/test_process_manager_unit.py`
- `tests/ui/test_main_window_shell.py`
- `tests/unit/test_runtime_telemetry.py`

Validates pre-decimation, hard point budgets, latest-frame replacement,
submission rejection, preview throttling, and dropped-frame counters.

### Step 2 — MoGe-2

- `tests/unit/test_moge2_adapter.py`
- `tests/unit/test_geometry_projection.py`
- `tests/unit/test_geometry_filtering.py`
- `tests/unit/test_geometry_colors.py`
- `tests/unit/test_geometry_validity.py`
- `tests/unit/test_runtime_messages.py`
- `tests/unit/test_mock_adapter.py`

MoGe tests use an injected fake model; they do not access Hugging Face. Validate
direct points, metric depth, masks, normals, intrinsics, point stride, BGR→RGB,
CPU FP32 fallback, warm-up, unloading, and public anonymous loading behavior.

### Step 2.1 — Static-image lifecycle

- `tests/unit/test_image_source_lifecycle.py`
- `tests/unit/test_capture_sources.py`
- `tests/ui/test_view_content_states.py`
- `tests/ui/test_main_window_shell.py`
- `tests/ui/test_viewport_layout.py`

Validates CONNECTED/READY/RESULT READY, image resend without disk I/O, no CAM
FPS for images, continuous LIVE behavior for video/camera/RTSP, pause and stop
states, and preserved results.

### Step 3 — Runtime telemetry

- `tests/unit/test_runtime_telemetry.py`
- `tests/unit/test_performance_formatting.py`
- `tests/ui/test_performance_bar.py`
- `tests/ui/test_main_window_shell.py`

Validates rate windows, counters, end-to-end latency wording, VRAM semantics,
model load/warm-up details, and preserving final metrics after Stop.

### Step 3.1 — Viewport and render controls

- `tests/unit/test_temporal_smoothing.py`
- `tests/ui/test_viewport_controls.py`
- `tests/ui/test_viewport_layout.py`
- `tests/ui/test_settings_sidebar.py`

Validates Cursor/Orbit/Pan/Zoom modes, Fit, focus/fullscreen mode, immediate point
size and smoothing, RGB/depth/normal recoloring, density, background, grid, axes,
background removal, disabled Mesh/VSync placeholders, and temporal smoothing.

### Geometry export foundation

- `tests/unit/test_geometry_export.py`

Validates dependency-free binary PLY output and round-trip point/color fidelity.

## Manual validation matrix

Run automated tests first, then perform these checks on Ubuntu with the normal
Qt platform (do not set `QT_QPA_PLATFORM=offscreen`).

### Static image / MoGe-2

- Select IMAGE, Browse, Connect, Start.
- Header: MOGE-2; model badge: METRIC READY.
- Source: CONNECTED; original panel: READY.
- Final state: RESULT READY.
- CAM FPS/INF FPS/RENDER FPS remain `—` for one result.
- Depth is coherent and point cloud is not mirrored or inverted.
- Run Start again; no checkpoint redownload or unnecessary model reload.
- Change RGB/Depth/Normals, Density, Point Size, and Smoothing without rerunning AI.

### Video

- Original panel and viewport remain LIVE.
- Preview stays near source FPS.
- Inference/render rates stabilize without growing delay.
- Pause → PAUSED; Resume → LIVE; Stop preserves last result and metrics.

### Camera

- Test mirror and each supported rotation.
- Disconnect releases the device so another program can open it.

### RTSP

- Test correct URL and an invalid URL.
- Confirm password is never printed in logs or persisted.
- Disconnect during reconnect backoff must return promptly.
- Preview stays responsive while 3D uses latest-frame-only updates.

### Viewport/OpenGL

- Cursor locks interaction.
- Orbit left-drag rotates.
- Pan left-drag translates.
- Zoom wheel/drag zooms.
- Fit Scene frames the cloud.
- Fullscreen/focus mode hides chrome; Esc restores it.
- Grid and axis remain aligned after resize.

### CUDA / RTX 2060

- Auto selects MoGe-2 ViT-S Normal, Efficient, FP16.
- Repeated Start/Stop does not increase VRAM continuously.
- No OOM for static image and configured live point budget.

### RTX 5090

- Auto selects Ultra and ViT-L only after the installed PyTorch build is proven
  compatible with the GPU architecture.
- Compare FP16/BF16 later during hardware tuning; do not assume BF16 is faster.

## Clean checkpoint rule

Only commit after all automated tests pass and the relevant manual source matrix
has been checked:

```bash
git status
./scripts/run_phase5_tests.sh full
git add -A
git commit -m "test: consolidate phase 5 regression suite"
```
