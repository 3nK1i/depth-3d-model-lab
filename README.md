# EIA TECH - Depth & 3D Model Lab

A PyQt6 desktop application for live monocular geometry estimation and interactive 3D point-cloud visualization on Ubuntu 24.04.

## Current status

Phases 1-4 and Phase 5 Steps 1-4.1 are implemented:

- Image, video, camera and RTSP capture
- Process-isolated inference worker with session-based stale-result rejection
- Latest-frame live-stream backpressure
- Hardware-aware Efficient, Balanced and Ultra profiles
- Development mock adapter for lifecycle diagnostics
- MoGe-2 metric depth, direct point maps, normals, mask and intrinsics
- Interactive VisPy point-cloud viewport
- End-to-end, model, worker, geometry and VisPy update timing
- Sliding-window capture, inference and 3D update rates
- Explicit CUDA allocated, reserved, peak and device-wide VRAM telemetry
- Model load/warm-up timing and latest-frame drop/backpressure diagnostics
- Functional Select, Orbit, Pan, Zoom, Fit and viewport-focus controls
- Immediate RGB/depth/normal recoloring and point-style updates from cached results
- Live temporal smoothing, background removal and immediate render-setting updates
- Functional floating Original Source inspection toolbar with save, rotate, fit, grid and focus actions
- Synchronized Depth Map colormap/range controls with panel focus mode

## Requirements

- Ubuntu 24.04 LTS
- Python 3.11 or 3.12
- NVIDIA driver and CUDA-compatible PyTorch for GPU inference

## Base installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## MoGe-2 installation

Install the official model package separately so the base UI remains usable without model dependencies:

```bash
python -m pip install -r requirements/models-moge2.txt
```

The first MoGe-2 start downloads the selected checkpoint from Hugging Face. The automatic RTX 2060 profile selects `Ruicheng/moge-2-vits-normal`; larger GPUs resolve to the Balanced or Ultra variant.

## Hardware check

```bash
python scripts/check_hardware.py
```

## Run

```bash
./scripts/run.sh
# or
python main.py
```

Select a source, choose **MoGe-2** in the MODEL card, connect the source, then press **Start Inference**. The optional **Development Mock** remains available when testing the UI without model weights.

Hover over the footer metrics for exact definitions and timing/VRAM breakdowns. The **LATENCY** value is end-to-end from worker submission to the completed depth/3D update; the tooltip also shows the model forward pass, worker preprocessing/postprocessing, UI geometry time and source-to-render age. The **VRAM USAGE** value is PyTorch reserved memory, with allocated, peak-reserved and device-wide usage listed in its tooltip. The MODEL info button shows checkpoint, device, precision, token count, load time and warm-up time.

## Viewport controls

The left viewport toolbar provides explicit **Select**, **Orbit**, **Pan**, **Zoom**, **Fit** and **Maximize viewport** actions. Press `Esc` to leave viewport-focus mode.

The floating render card updates an already displayed static result without rerunning MoGe-2:

- **RGB Points**, **Depth Colors** and **Surface Normals** recolor immediately
- **Point Size** and marker **Smoothing** update the current VisPy visual immediately
- **Density** resamples the cached point cloud immediately
- **Reset View** restores and fits the camera

**Mesh (Poisson)** remains disabled until the dedicated reconstruction phase adds threaded mesh generation. **VSync** is enabled when the VisPy canvas is created and is shown as a locked setting because changing the swap interval would require recreating the OpenGL surface.

The right sidebar's colormap, depth range, edge filtering, background removal, background color, grid, axes and point budget all apply immediately to the cached result. Temporal smoothing affects only consecutive video, camera and RTSP frames; static images are never temporally blended.

## Source and depth inspection controls

The **Original Source** toolbar is overlaid on the media surface and contains only real actions:

- save the current frame
- rotate the preview clockwise (display only)
- focus the Source panel, with `Esc` to restore the full interface
- toggle fit-to-panel versus 1:1 display
- toggle a rule-of-thirds inspection grid

The **Depth Map** footer mirrors the sidebar colormap, minimum depth and maximum depth values. Changes made in either location update the other and immediately rerender the cached depth/geometry result. The redundant settings-gear placeholder has been removed; the remaining fullscreen action focuses the Depth panel and exits with `Esc`.

Public MoGe-2 checkpoints continue to load anonymously with `token=False`. The application suppresses only Hugging Face's harmless anonymous-rate-limit advisory while preserving real network, repository, cache and authentication errors.

## Model lifecycle

The model that lives in the spawned worker has an explicit lifecycle independent
of source/runtime state:

```text
UNLOADED → LOADING → WARMING UP → READY → UNLOADING → UNLOADED
                         ↘ FAILED ↗
```

When inference is stopped or a static result is ready, the MODEL and GPU Profile
controls are unlocked. Changing the model or a load-affecting profile setting
performs an acknowledged unload followed by a new load/warm-up in the same
worker process. The connected image/video/camera/RTSP source is preserved.
Per-frame/render settings such as Point Budget do not reload model weights.

MoGe-2 unload synchronizes pending CUDA work, releases model references, runs
garbage collection, clears the PyTorch allocator cache and returns cleanup
telemetry to the UI process. Development Mock follows the same lifecycle
contract so switching can be tested without CUDA or model downloads.

## Tests

```bash
QT_QPA_PLATFORM=offscreen pytest -q
```

## Project structure

```text
app/                 Qt UI and application controllers
core/capture/        Image, video, camera and RTSP sources
core/inference/      Worker process and IPC manager
core/models/         Unified model adapters and model catalog
core/geometry/       Depth, direct points, filtering and sampling
core/runtime/        State machine, sessions and watchdog
requirements/        Optional model dependency groups
tests/               Unit, UI and integration tests
```
