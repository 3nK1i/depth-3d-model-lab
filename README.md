# EIA TECH - Depth & 3D Model Lab

A PyQt6 desktop application for live monocular geometry estimation and interactive 3D point-cloud visualization on Ubuntu 24.04.

## Current status

Phases 1-4 and Phase 5 Steps 1-5 are implemented:

- Image, video, camera and RTSP capture
- Process-isolated inference worker with session-based stale-result rejection
- Latest-frame live-stream backpressure
- Hardware-aware Efficient, Balanced and Ultra profiles with total/free-VRAM safety gates
- Automatic RTX 2060 and RTX 5090 tuning with no source or config edits
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
- Structured failure classification with bounded worker crash/timeout/OOM recovery
- Repeated lifecycle, restart, shutdown and memory-stability stress tooling
- Simulated cross-GPU tuning matrix for hardware that is not currently available

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

## Hardware check and automatic tuning

Inspect the current machine:

```bash
python scripts/check_hardware.py --assert-ready
```

The resolver uses CUDA readiness, compute capability, total VRAM and startup
free VRAM. It does not require a product-specific config file. The default
tiers are:

- **RTX 2060 / 6 GB class:** Efficient, ViT-S, FP16, level 5, 250K static / 100K live points
- **8-19 GB class:** Balanced, ViT-B, FP16, level 7, 500K static / 250K live points
- **RTX 5090 / >=20 GB class:** Ultra, ViT-L, FP16, level 9, 1M static / 500K live points

FP16 remains the automatic high-end default because it is the established MoGe
execution path. BF16 support is detected and reported but is not selected
implicitly. TF32 is enabled only on supported Balanced/Ultra CUDA devices, and
native SDPA is enabled when the installed MoGe version exposes it.

The RTX 5090 path can be validated without owning the card:

```bash
python scripts/check_hardware.py --simulate rtx2060
python scripts/check_hardware.py --simulate rtx5090
python scripts/check_hardware.py --simulate rtx5090-pressure
```

The pressure simulation proves that a 32 GB device temporarily falls back to
Balanced when less than 16 GB is free at startup. When the same repository is
launched later on an RTX 5090 with a compatible CUDA-enabled PyTorch build, Auto
selects Ultra without editing code or configuration.

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

## Phase 5 hardware tuning

Profile tuning is serialized into the worker load command and recorded in the
MODEL tooltip. The worker applies one or two profile-specific warm-up passes,
`torch.inference_mode()`, safe float32 matmul precision, optional TF32 and
optional native SDPA. Point-cloud rendering remains independently capped for
live sources so a fast GPU cannot overwhelm the Qt/VisPy UI thread.

Auto selection also considers startup free VRAM. A manual tier that exceeds the
current safe memory envelope is downgraded with a `SAFE FALLBACK` badge instead
of predictably producing an OOM. Unknown future NVIDIA GPUs follow the same
capability and memory rules rather than a hard-coded product-name list.

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


## Failure recovery

Worker/model failures are classified into stable categories instead of being
reported as undifferentiated exception strings. The UI distinguishes CUDA OOM,
missing dependencies, offline/uncached checkpoints, missing or corrupted model
files, authentication problems, invalid configuration, warm-up failure, worker
timeout and worker crash.

Recoverable failures preserve the configured image/video/camera/RTSP source and
unlock model controls. CUDA OOM and dead/unresponsive workers trigger one
bounded worker restart; the application never loops indefinitely. A crashed
worker may reload the previously resident model, but inference is not restarted
automatically. A timed-out model load returns to a retryable source state so the
user can reconnect, choose a safer profile, or switch to Development Mock.

The model-load watchdog uses a longer timeout than Stop/Unload so first-time
checkpoint downloads and warm-up do not create false worker failures. A
separate result watchdog detects a submitted frame that never returns, then
terminates and recreates only the inference worker while preserving the source.

## Lifecycle stress and shutdown hardening

Worker restarts now use fresh pipes and latest-only queues, reap dead children,
and escalate from cooperative shutdown to terminate/kill only when a worker does
not exit. Every spawn has a diagnostic generation number and the process manager
exposes the current worker PID and restart count for reports. Intentional
shutdown suppresses crash callbacks and is safe while the application is
loading, warming up, running, paused, switching models or recovering a worker.
Late acknowledgements and results are ignored after shutdown starts. The camera
device probe is cooperatively cancelled and joined during application close, so a
fast close cannot leave a capture-enumeration thread behind.

The longer stress pack is deliberately separate from the normal regression
suite:

```bash
./scripts/run_phase5_tests.sh stress
./scripts/run_phase5_tests.sh all
```

Dependency-free Development Mock lifecycle loop:

```bash
python scripts/run_lifecycle_stress.py --cycles 10 --frames-per-cycle 2
```

Repeated forced worker-crash/restart loop:

```bash
python scripts/run_worker_recovery_stress.py --cycles 5
```

Memory stability with Development Mock:

```bash
python scripts/run_memory_stress.py --model mock --cycles 10
```

Real MoGe-2 load/infer/unload VRAM validation on the detected safe profile:

```bash
python scripts/run_memory_stress.py \
  --model moge2 \
  --profile auto \
  --cycles 5
```

Add `--strict-device-vram` only when other GPU processes are stable. JSON reports
are written under `outputs/stress/`. Device-wide VRAM may vary when
Xorg, GNOME, browsers or other CUDA applications are active; allocator reserved
memory after unload and repeated tail drift are the stronger model-lifecycle
signals.

## Tests

```bash
./scripts/run_phase5_tests.sh smoke
./scripts/run_phase5_tests.sh unit
./scripts/run_phase5_tests.sh ui
./scripts/run_phase5_tests.sh integration
./scripts/run_phase5_tests.sh full
```

`full` keeps the regular development loop fast and excludes `tests/stress`. Use
`stress` for the repeated process loops or `all` for the complete pack.

## Project structure

```text
app/                 Qt UI and application controllers
core/capture/        Image, video, camera and RTSP sources
core/inference/      Worker process and IPC manager
core/models/         Unified model adapters and model catalog
core/geometry/       Depth, direct points, filtering and sampling
core/runtime/        State machine, sessions and watchdog
requirements/        Optional model dependency groups
tests/               Unit, UI, integration and explicit stress tests
```
