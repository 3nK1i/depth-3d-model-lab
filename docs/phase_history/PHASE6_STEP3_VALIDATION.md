# Phase 6 Step 3 — UniDepthV2 Validation

Phase 5 and the completed Phase 6 Steps 1–2 remain the compatibility baseline. Do not commit until compile validation, the quick gate, the full gate, and the applicable manual scenarios all pass.

## Installation validation

```bash
conda activate depth
cd /home/eiatech/Desktop/depth_3d_model_lab

python scripts/install_unidepth_v2.py
python scripts/download_unidepth_v2_models.py --variant auto
```

The installer pins the official UniDepth source to commit
`8d8cfe4c7ee15297099983607febf0d4f32eb3d6`. Model snapshots are also pinned
to reviewed Hugging Face revisions. The adapter installs an isolated runtime
padding guard for a known upstream rounding edge case; the model architecture
and weights are unchanged.

Expected automatic download:

- RTX 2060 6 GB: `checkpoints/unidepth-v2/vits14`
- 8–19 GB CUDA GPU: `checkpoints/unidepth-v2/vitb14`
- RTX 5090 32 GB: `checkpoints/unidepth-v2/vitl14`

Verify:

```bash
python -c "from unidepth.models import UniDepthV2; print('UniDepthV2 import OK')"
find checkpoints/unidepth-v2 -maxdepth 2 -type f | sort
cat checkpoints/unidepth-v2/.ready
cat third_party/UniDepth/.eia_unidepth_v2_compat.json
```

Optional: download every registry variant for cross-profile testing:

```bash
python scripts/download_unidepth_v2_models.py --variant all
```

## Compile and automated tests

```bash
python -m compileall -q app core tests scripts main.py
./scripts/run_phase6_step3_tests.sh quick
./scripts/run_phase6_step3_tests.sh full
```

The full gate must run Unit, UI, Integration, and Stress suites in separate pytest processes.

## Manual validation matrix

### 1. Startup and registry

- Start with `python main.py`.
- Confirm the model order is: MoGe-2, Depth Pro, UniDepthV2, Development Mock.
- Confirm MoGe-2 remains the default after a clean settings reset.
- Select UniDepthV2 and restart the app; confirm selection persistence.
- Confirm the header badge reads `UNIDEPTH V2` when loaded.
- Confirm the active model badge reads `METRIC + UNCERTAINTY`.
- Confirm no UniDepth/PyTorch model is imported merely by opening the model combo.

### 2. Automatic hardware profiles

#### RTX 2060 6 GB

- Auto profile selects `UniDepthV2 ViT-S/14`.
- Inference detail shows `Level 2 · Metric + uncertainty`.
- Precision shows FP16.
- Static point budget is 250K; live point budget is 100K.
- Model loads without CUDA OOM and the UI remains responsive.

#### RTX 5090 32 GB

- Auto profile selects `UniDepthV2 ViT-L/14`.
- Inference detail shows `Level 8 · Metric + uncertainty`.
- Precision shows FP16.
- Static point budget is 1M; live point budget is 500K.
- TF32 is enabled only when supported by the installed PyTorch build.

#### Fallback behavior

- Under startup VRAM pressure, Auto downgrades to a safer profile.
- Manually requesting a profile above available VRAM produces the existing safe fallback.
- CPU fallback uses FP32 and remains functional, although slow.

### 3. Image scenarios

Test at least:

- Landscape RGB image.
- Portrait image.
- Square image.
- Small image below 256 pixels on one side.
- Large image above Full HD.
- Very wide image.
- Very tall image.
- Exact regression image size `500×206` (width × height).
- Image with indoor geometry.
- Image with outdoor geometry.
- Low-texture walls/floors.
- Reflective or dark regions.

For each image verify:

- RGB view is unchanged and has correct channel order.
- Depth output has the same width and height as the source.
- The `500×206` regression input does not collapse to a one-pixel dimension.
- Depth values are finite and positive in valid regions.
- Point cloud appears with plausible orientation and scale.
- Predicted intrinsics are accepted and reported as model-predicted.
- Point count respects the selected point budget.
- Latency, precision, model name, resolution, and VRAM update.
- Stop, restart, and re-run produce stable results.

### 4. Video scenarios

- Load a short MP4 and run to end-of-video.
- Pause and resume repeatedly.
- Stop mid-video and restart.
- Seek/reconnect using the existing source workflow.
- Switch to UniDepthV2 before inference and while stopped.
- Verify frame replacement/backpressure keeps the UI responsive.
- Verify temporal smoothing remains controlled by the existing generic pipeline.
- Run at least 10 minutes and check bounded CPU/GPU memory.

### 5. USB camera scenarios

- Connect camera, start, pause, resume, and stop.
- Disconnect while stopped and while paused.
- Change camera resolution if supported.
- Cover/uncover the lens and verify recovery.
- Verify live point budget is used.
- Verify no stale depth frame remains after reconnecting.

### 6. RTSP scenarios

- Connect to a valid RTSP stream and start inference.
- Pause/resume and stop/restart.
- Disconnect/reconnect normally.
- Interrupt the network briefly and verify source/worker recovery.
- Try an invalid RTSP URL and verify a controlled source error.
- Run for at least 10 minutes and check memory stability.

### 7. Model lifecycle and switching

Run these sequences with the source kept connected:

- MoGe-2 → UniDepthV2 → MoGe-2.
- Depth Pro → UniDepthV2 → Depth Pro.
- UniDepthV2 → Development Mock → UniDepthV2.
- MoGe-2 → Depth Pro → UniDepthV2 → Mock → MoGe-2.

Repeat the full cycle 10 times. Verify:

- Correct header badge, variant, precision, latency, and point count after every switch.
- Previous model CUDA memory is released.
- VRAM returns near the same baseline on repeated cycles instead of increasing.
- The source remains connected.
- No stale result from the previous model is displayed.
- No worker restart loop occurs.

### 8. Output semantics

- Depth is marked metric.
- Direct UniDepthV2 points are used rather than re-projecting depth unnecessarily.
- Intrinsics are marked `model_predicted` when returned.
- If intrinsics are absent, the generic fallback estimates `fx`, `fy`, `cx`, `cy` and marks them `estimated`.
- Confidence is carried as `uncertainty`, not as absolute probability.
- Lower uncertainty should rank more reliable regions only within the same input; do not compare absolute values across unrelated images.
- Invalid or non-finite depth/uncertainty pixels are excluded by the validity mask.
- No calibrated-measurement claim appears anywhere.

### 9. Render and cached-result controls

Without rerunning inference, change:

- Colormap.
- Min/max depth.
- Edge filtering.
- Background removal.
- Point size and density.
- Grid and axes.
- Point budget.
- RGB/depth render modes that are currently supported.
- Reset View, fit, orbit, pan, and zoom.

Verify cached render updates do not invoke UniDepthV2 again unless the point-budget path explicitly requires a new result.

### 10. Failure and recovery scenarios

#### Missing runtime

Temporarily make `unidepth` unavailable. Selecting/loading UniDepthV2 must show a clear installation error and leave MoGe-2, Depth Pro, and Mock usable.

#### Missing snapshot

Temporarily rename the selected variant directory. Loading must report the exact missing directory and the downloader command.

#### Incomplete snapshot

Temporarily remove `config.json` or the weight file. Loading must fail cleanly without crashing the worker.

#### Wrong variant available

On RTX 2060, keep only ViT-L and remove ViT-S. Auto must not silently load the unsafe model; it must report the missing ViT-S snapshot.

#### CUDA OOM

Create GPU memory pressure and load/run UniDepthV2. Verify one controlled OOM recovery, CUDA cleanup, no restart loop, and continued availability of Mock or a lighter model.

#### Worker failure and timeout

Use the existing test hooks to simulate a worker crash and an inference timeout. Verify existing Phase 5 recovery behavior remains unchanged.

### 11. Shutdown scenarios

Close the application while:

- Idle.
- Model loaded but inference stopped.
- Image inference completed.
- Video/camera/RTSP inference running.
- Inference paused.
- A controlled load error is displayed.

After each case verify no worker remains:

```bash
pgrep -af "depth_3d_model_lab|core.inference.worker"
```

### 12. License gate

UniDepth and its model weights are published under CC BY-NC 4.0. Keep the registry status `noncommercial_review_required`. Do not approve commercial deployment until EIA TECH has completed legal review or obtained appropriate permission.

## Git review and commit

```bash
git status --short
git diff --stat
git add -A
git commit -m "feat(models): integrate UniDepthV2 metric geometry adapter"
```
