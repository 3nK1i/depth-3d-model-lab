# EIA TECH - Depth & 3D Model Lab

A PyQt6 desktop application for live monocular depth estimation, measurement and inspection, and persistent 3D reconstruction on Ubuntu 24.04.

## Current status

Version `0.8.0` is the **Phase 8 Reconstruction Stable** release. The release commit is qualified before the annotated `v0.8.0` tag is created in the final 8.9I3 release step.

The `v0.8.0` release includes:

- Live 3D for Image, Video, USB camera and RTSP sources
- Interactive Benchmark, Snapshot and Stable Video operational modes
- Process-isolated, model-agnostic inference with bounded crash, timeout and OOM recovery
- Automatic capability-driven GPU profiles from RTX 2060-class hardware through the qualified high-end path
- MoGe-2, Depth Pro, UniDepthV2, Metric3D v2, Depth Anything V2, Video Depth Anything and Marigold
- YOLO instance segmentation, prompted SAM2 refinement, depth/perception alignment and object-specific 3D extraction
- Estimated 3D boxes, measurements, uncertainty, temporal stabilization and Measure & Inspect workflows
- Dedicated Reconstruction workspace with live SLAM and offline SfM workflows
- Canonical geometry, geometry quality/uncertainty, surface reconstruction, semantic context and inspection/QC
- Gaussian splatting and NeRF/Nerfacto appearance representations behind isolated external backend boundaries
- Reconstruction session persistence, checkpoint/resume, representation switching, recovery and authority-aware export
- Reconstruction fullscreen/focus interaction and responsive UI behavior across the existing operational modes
- Final current-tree, backward-compatibility, RTSP and combined-UI qualification gates

Measurements remain engineering estimates rather than certified metrology unless a separate qualified metric reference/calibration path is supplied. The calibrated known-size campaign is deliberately post-`v0.8.0` work.

## Deferred calibrated known-size validation

The known-size benchmark tooling is retained for a post-`v0.8.0` calibrated validation campaign. It is not required for the engineering-estimate measurement workflow, and its presence does not make the current product certified metrology.

Create an editable benchmark manifest when that campaign begins:

```bash
python scripts/create_known_size_benchmark_template.py \
  --output outputs/phase7_5/known_size_benchmark.json
```

After adding real exports and independently measured reference dimensions, run:

```bash
python scripts/run_known_size_validation.py \
  --manifest outputs/phase7_5/known_size_benchmark.json \
  --output-dir outputs/phase7_5/known_size_validation
```

See `KNOWN_SIZE_BENCHMARK_GUIDE.md` and `TEST_STRATEGY_PHASE7_5_RELEASE.md` for the retained validation methodology and historical release-gate design.

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

## Optional Phase 6 model runtimes

The base UI remains usable without optional models. Install and download only the models required on the machine, using the dedicated scripts and requirement files under `requirements/`. Local third-party sources and checkpoints are excluded from Git.

Registered production models:

- MoGe-2
- Depth Pro
- UniDepthV2
- Metric3D v2
- Depth Anything V2
- Video Depth Anything
- Marigold

## Hardware check and automatic tuning

Inspect the current machine:

```bash
python scripts/check_hardware.py --assert-ready --require-cuda
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

Hover over the footer metrics for exact definitions and timing/VRAM breakdowns. The **LATENCY** value is end-to-end from worker submission to the completed depth/3D update; the tooltip also shows the model forward pass, worker preprocessing/postprocessing, UI geometry time and source-to-render age. The **MODEL VRAM** value is PyTorch reserved memory, with allocated, peak-reserved and device-wide usage listed in its tooltip. The MODEL info button shows checkpoint, device, precision, token count, load time and warm-up time.


## Phase 7 operational modes

### Interactive Benchmark

The Benchmark tab runs one or more registry models in isolated processes and displays load time, warm-up, latency, FPS, model VRAM, output resolution, depth scale and intrinsics provenance. Reports are written as JSON, CSV and Markdown.

### Snapshot

Snapshot captures an independent copy of the latest source frame, runs one selected model outside the Live 3D worker and atomically exports RGB, numerical depth, preview depth, point cloud, supported confidence/normals/mask arrays and a manifest. Live inference must be stopped, but the source may remain connected.

### Stable Video

Stable Video supports Video, USB camera and RTSP. Temporal models preserve their native sequence state; stateless models use generic smoothing. Latest-frame backpressure prevents queue growth, Pause/Resume preserves the sequence, and Stop resets it. Finalized runs export synchronized RGB/depth MP4 files, per-frame performance CSV, a manifest and optional numerical depth frames.

## Phase 8 Reconstruction

Reconstruction is a dedicated application workspace rather than a separate top-level SfM/SLAM product. It coordinates live and offline reconstruction while preserving the existing Live 3D, Benchmark, Snapshot, Stable Video and Measure & Inspect workflows.

- **Live SLAM:** continuous Video, USB camera and RTSP capture through an isolated native stella worker, with trajectory, sparse-map/keyframe and diagnostics presentation plus bounded recovery.
- **Offline SfM:** image-sequence and recorded-video preparation with isolated PyCOLMAP execution, workspace contracts, resume support and reconstruction-quality inspection.
- **Geometry:** camera-geometry and scale/reference abstractions, dense multi-view fusion, geometry quality/uncertainty, surface mesh generation and persistent canonical scene geometry.
- **Semantics and QC:** semantic object context, reconstruction-side measurement presentation, measurement validity, inspection/QC and reference-comparison workflows.
- **Appearance:** Gaussian splatting and NeRF/Nerfacto outputs are integrated as isolated appearance representations and can be persisted and switched alongside canonical geometry.
- **Persistence and export:** reconstruction sessions, checkpoints, reload/reproducibility, product export bundles, QC reports and SHA-256-backed provenance.

Depth Anything 3 is integrated through an isolated external provider for supporting depth/geometry workflows; heavy reconstruction, Gaussian and NeRF runtimes remain outside the production `depth` environment.

### Reconstruction authority and limitations

The release keeps explicit authority boundaries:

- canonical geometry is the authoritative scene geometry
- semantic geometry supplies semantic/object context but does not replace canonical geometry
- measurement results remain measurement evidence with explicit uncertainty and rejection/validity state
- inspection/QC remains the inspection authority
- Gaussian and NeRF/Nerfacto outputs are **appearance only** and are never metric, canonical, measurement, QC or reference authority
- `certified_metrology=false` remains the correct product posture unless a separate qualified calibration/reference process establishes otherwise
- monocular live SLAM or monocular reconstruction may remain in arbitrary scale without a valid metric reference; camera intrinsics alone do not establish metric scale

`v0.8.0` does not claim the future calibration/metric-qualification workflow, a broader custom Auto Segment workspace, a unified digital-twin workspace, or a fully interactive neural free-view renderer. Existing YOLO/SAM2 measurement adapters remain available, while those larger workflows are reserved for post-`v0.8.0` development.


## Viewport controls

The left viewport toolbar provides explicit **Select**, **Orbit**, **Pan**, **Zoom**, **Fit** and **Maximize viewport** actions. Press `Esc` to leave viewport-focus mode.

The floating render card updates an already displayed static result without rerunning MoGe-2:

- **RGB Points**, **Depth Colors** and **Surface Normals** recolor immediately
- **Point Size** and marker **Smoothing** update the current VisPy visual immediately
- **Density** resamples the cached point cloud immediately
- **Reset View** restores and fits the camera

**Mesh (Poisson)** remains disabled in the shared Live 3D render card; Phase 8 surface-mesh processing belongs to the dedicated Reconstruction workflow and does not change that legacy control. **VSync** is enabled when the VisPy canvas is created and is shown as a locked setting because changing the swap interval would require recreating the OpenGL surface.

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


## Phase 6 benchmark and release gate

Run the cross-model benchmark on the current GPU:

```bash
python scripts/run_phase6_benchmark.py \
  --models all \
  --profile auto \
  --point-budget 100000 \
  --require-all
```

Reports are written under `outputs/phase6/`. Each production model runs in a fresh Python process so CUDA allocators, optional dependencies and temporal caches cannot contaminate the next measurement.

Run the quick release gate while developing:

```bash
./scripts/run_phase6_release.sh quick
```

Run the complete release gate after generating the benchmark:

```bash
./scripts/run_phase6_release.sh full --require-benchmark
```

After committing, add `--require-clean-git`. The atomic release report is written to:

```text
outputs/release/phase6_release_report.json
```

The same benchmark command is used on RTX 2060 and RTX 5090; Auto selects the appropriate model variants and compute profile.

## Diagnostics

Normal startup writes rotating logs to:

```text
logs/app.log
logs/worker.log
```

The two processes use separate files to avoid cross-process rotation conflicts.
Uncaught application/thread exceptions and full worker model tracebacks are
recorded there. `DEPTH3D_LOG_DIR` and `DEPTH3D_LOG_LEVEL` can override the
location and level without editing the repository.

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

The final Phase 8 pre-H release-candidate requalification remains available as historical qualification evidence:

```bash
./scripts/run_phase8_9_pre_h_final_requalification_tests.sh quick
./scripts/run_phase8_9_pre_h_final_requalification_tests.sh full
```

The `v0.8.0` metadata transition is qualified by the dedicated final-release gate:

```bash
./scripts/run_phase8_9i2_tests.sh quick
./scripts/run_phase8_9i2_tests.sh affected
./scripts/run_phase8_9i2_tests.sh full
```

The final-release gate validates the coordinated `0.8.0` identity, documentation and authority boundaries, then runs the current unit/UI/integration/stress suites without restarting the expensive real SfM/Gaussian/NeRF campaigns.

## Project structure

```text
app/                 Qt UI and application controllers
core/capture/        Image, video, camera and RTSP sources
core/inference/      Worker process and IPC manager
core/models/         Registry, normalized contracts and model adapters
core/benchmark/      Deterministic assets and benchmark/report engine
core/geometry/       Depth, direct points, filtering and sampling
core/reconstruction/ Reconstruction contracts, geometry, SLAM/SfM, persistence, QC and appearance orchestration
core/runtime/        State, sessions, watchdog, logs and release diagnostics
native/              Isolated native reconstruction workers
requirements/        Optional model and reconstruction dependency groups
docs/                Phase history and release/test strategy evidence
tests/               Unit, UI, integration, stress and release tests
```

Measurements are engineering estimates; calibrated known-size and metric qualification remains post-`v0.8.0` work.