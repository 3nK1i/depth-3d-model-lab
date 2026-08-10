# Phase 6 Step 2 — Depth Pro Validation

Phase 5 and Phase 6 Step 1 remain the compatibility baseline. Do not commit
until compile, quick, full, and manual validation all pass.

## Installation

```bash
python -m pip install -r requirements/models-depth-pro.txt
python scripts/download_depth_pro_checkpoint.py
```

The checkpoint must exist at `checkpoints/depth_pro.pt` and must remain
untracked by Git.

## Compile and automated gates

```bash
python -m compileall -q app core tests scripts main.py
./scripts/run_phase6_step2_tests.sh quick
./scripts/run_phase6_step2_tests.sh full
```

The full gate must finish without failures, skips caused by missing project
dependencies, native crashes, or lingering worker processes.

## Manual validation matrix

### 1. Startup and registry

- Launch with `python main.py`.
- Confirm MoGe-2 remains the default selection.
- Confirm Depth Pro appears between MoGe-2 and Development Mock.
- Select Depth Pro and confirm `Depth Pro · DINOv2-L`.
- Confirm inference detail is `1536² internal · Metric`.
- Confirm RTX 2060 Auto resolves to Efficient and FP16.
- Confirm RTX 5090 Auto resolves to Ultra and FP16.
- Confirm changing models does not import or load weights before Start.

### 2. Image source

- Connect a landscape image, a portrait image, and a small image.
- Start Depth Pro and confirm RGB, depth, and point cloud appear.
- Confirm depth is metric and visually plausible from near to far.
- Confirm the point cloud uses the predicted focal length without stretching.
- Test RGB Colors and Depth Colors render modes.
- Confirm Surface Normals is unavailable because Depth Pro does not return normals.
- Change point budget and confirm rendering updates without rerunning inference.
- Change colormap, min/max depth, filtering, background, grid, and axes.
- Stop, restart, unload, and reload using the same connected image.

### 3. Video source

- Connect a short and a long video.
- Start, pause, resume, stop, and restart Depth Pro.
- Confirm latest-frame backpressure keeps the UI responsive.
- Confirm end-of-video handling remains correct.
- Seek or reconnect according to the existing Phase 5 video workflow.
- Confirm repeated stop/start cycles do not increase CPU or GPU memory continuously.

### 4. USB camera

- Connect the camera and start Depth Pro.
- Confirm live frames, depth, geometry, FPS, and latency update.
- Pause and resume without reconnecting the camera.
- Stop and restart while the camera stays connected.
- Disconnect while stopped and while running.
- Confirm camera unavailability or unplugging produces a controlled source error.

### 5. RTSP camera

- Connect a valid RTSP stream and run Depth Pro.
- Confirm the UI stays responsive despite Depth Pro's heavier inference.
- Pause, resume, stop, restart, and disconnect.
- Interrupt the network briefly and confirm the source/runtime fails safely.
- Reconnect the stream without restarting the application.

### 6. Model switching

With one source connected, validate:

- MoGe-2 → Depth Pro
- Depth Pro → MoGe-2
- Depth Pro → Development Mock
- Development Mock → Depth Pro
- At least ten repeated model switches

For every switch, confirm the previous model unloads, the source remains
connected, stale results are rejected, and the new model owns the footer and
viewport metadata.

### 7. Hardware and profiles

On RTX 2060:

- Auto selects Efficient, CUDA, and FP16.
- Manual Balanced or Ultra follows the existing safe-fallback rules.
- Loading does not create an unrecoverable transient FP32 CUDA allocation.
- If OOM occurs, the worker recovers and the source remains usable.

On RTX 5090:

- Auto selects Ultra, CUDA, and FP16.
- Higher static/live point budgets are applied automatically.
- TF32 backend tuning is enabled only when supported.

### 8. Telemetry and output contract

- Footer model name is Depth Pro.
- Precision is FP16 on CUDA and FP32 on CPU.
- Load, warm-up, inference, preprocessing, and worker timings are populated.
- Allocated, reserved, peak, device-used, and total VRAM remain plausible.
- Point count follows the selected static or live point budget.
- Depth and validity-mask dimensions match the source frame.
- Predicted intrinsics are used when focal length is valid.
- Estimated intrinsics fallback remains labelled `estimated` when prediction is invalid.

### 9. Failure scenarios

Checkpoint missing:

```bash
mv checkpoints/depth_pro.pt checkpoints/depth_pro.pt.disabled
python main.py
mv checkpoints/depth_pro.pt.disabled checkpoints/depth_pro.pt
```

Confirm a clear checkpoint error, no worker restart loop, and successful retry
after restoration.

Also validate:

- Missing Depth Pro dependency reports the requirement file.
- Corrupted checkpoint reports a controlled load failure.
- Invalid source frames do not crash the worker.
- OOM recovery does not disconnect the source.
- Stop and application shutdown work after a failed load.

### 10. Shutdown and stability

- Run Depth Pro for at least 10 minutes on video or RTSP.
- Perform repeated pause/resume and model switches.
- Close the application while idle, running, paused, and after an error.
- Confirm no inference worker remains with `pgrep -af depth_3d_model_lab`.
- Confirm logs contain no repeated crash/restart loop.

## Commit

```bash
git status --short
git add -A
git commit -m "feat(models): integrate Depth Pro metric depth adapter"
```
