# Phase 6 Step 5 Validation — Depth Anything V2 + Video Depth Anything

This checklist validates the two registry-driven integrations without changing the stable Phase 5 runtime architecture.

- **Depth Anything V2:** frame-based **relative** depth.
- **Video Depth Anything:** stateful, streaming **metric** depth with **estimated camera intrinsics**.
- Calibration and calibrated measurements remain postponed.
- The upstream Video Depth Anything streaming path is experimental and must be treated as such during evaluation.

Do not commit Step 5 until compile validation, the quick gate, the full regression gate, and all applicable manual scenarios pass.

## 1. Installation and asset checks

```bash
conda activate depth
cd /home/eiatech/Desktop/depth_3d_model_lab

python scripts/install_depth_anything_models.py
python scripts/download_depth_anything_models.py --family all --variant auto
```

Expected on RTX 2060:

- Depth Anything V2 Small checkpoint
- Metric Video Depth Anything Small checkpoint

Expected on RTX 5090:

- Depth Anything V2 Large checkpoint
- Metric Video Depth Anything Large checkpoint

Verify:

```bash
cat third_party/Depth-Anything-V2/.eia_depth_anything_v2_ready.json
cat third_party/Video-Depth-Anything/.eia_video_depth_anything_ready.json
cat checkpoints/depth-anything-v2/.ready
cat checkpoints/video-depth-anything/.ready

find checkpoints/depth-anything-v2 -maxdepth 1 -type f -ls
find checkpoints/video-depth-anything -maxdepth 1 -type f -ls
```

Pass criteria:

- Both official source imports report `OK` during installation.
- The existing PyTorch version is unchanged.
- Each source marker records an exact Git commit.
- Each checkpoint is complete and stored under the expected directory.
- No source tree or checkpoint appears in `git status`.

### xFormers note

The upstream VDA repository pins an old Torch/xFormers stack. This integration deliberately preserves the project's validated CUDA PyTorch installation. An `xFormers not available` warning is acceptable if inference succeeds and memory remains stable; do not install the upstream pinned xFormers package blindly.

## 2. Compile and automated validation

```bash
python -m compileall -q app core tests scripts main.py
./scripts/run_phase6_step5_tests.sh quick
./scripts/run_phase6_step5_tests.sh full
```

The quick gate validates:

- Registry entries and ordering
- Hardware-aware model variants
- Relative and metric output contracts
- Estimated intrinsics
- BGR/RGB preprocessing contracts
- Point budgets and geometry outputs
- Streaming cache reset behavior
- STOP-triggered generic sequence reset
- Missing dependencies, sources, and checkpoints
- Corrupted or malformed outputs
- Installer and downloader behavior
- Repeated adapter lifecycle stress
- Backward compatibility with all existing models

The full gate runs unit, UI, integration, and stress suites in separate Python processes.

## 3. Startup and registry UI

Start the application:

```bash
python main.py
```

Verify the model order:

1. MoGe-2
2. Depth Pro
3. UniDepthV2
4. Metric3D v2
5. Depth Anything V2
6. Video Depth Anything
7. Development Mock

Verify:

- MoGe-2 remains the default after a clean settings reset.
- Both new models appear without importing their heavy runtimes at application startup.
- Selecting Depth Anything V2 shows a relative-depth badge and relative-depth description.
- Selecting Video Depth Anything shows the temporal metric badge.
- Model variant, inference detail, precision, and GPU profile update correctly.
- No existing model's metadata, badge, or settings are changed.

## 4. Depth Anything V2 — image scenarios

Test at least:

- Landscape image
- Portrait image
- Square image
- Very wide image
- Very tall image
- Small image
- Full-HD or larger image
- Image containing dark/textureless areas
- Image containing thin structures and sharp edges

For each image verify:

- Model loads and warms up successfully.
- Depth map appears at the source resolution.
- Near/far ordering is visually sensible.
- Point cloud orientation is not inverted.
- RGB point colors match the source.
- Camera orbit, pan, zoom, fit, and reset remain functional.
- Point-budget changes update the geometry without model reload.
- Header and footer telemetry update.
- Output is clearly treated as **relative**, not metric.
- No measurement or calibrated-scale claim appears.

## 5. Depth Anything V2 — Video, USB camera, and RTSP

For each source type:

1. Connect.
2. Start inference.
3. Pause.
4. Resume.
5. Stop.
6. Restart.
7. Disconnect.

Verify:

- Frames continue updating without stale results.
- Stop/restart does not require model reload.
- Source switching clears old frames and geometry.
- Video end-of-file is handled normally.
- Camera disconnect/reconnect is recoverable.
- RTSP temporary interruption follows the existing Phase 5 recovery behavior.
- Relative depth can flicker between frames; this is expected for the frame-based model and must not be presented as temporally stable.

## 6. Video Depth Anything — static image behavior

Use a static image as a one-frame sequence.

Verify:

- The first frame produces a metric depth map and point cloud.
- Intrinsics are marked `estimated`.
- Results are not described as calibrated.
- Stop then Start resets the sequence and produces a fresh first-frame result.
- Switching to another image resets temporal state.

## 7. Video Depth Anything — continuous video

Use a video with visible camera or object motion. Run for at least two minutes.

Verify:

- Temporal depth is visibly more stable than frame-based Depth Anything V2.
- The first result appears without waiting for a 32-frame batch.
- Sequential frames preserve temporal state.
- VRAM may rise while temporal caches are populated, then must reach a bounded plateau.
- Depth, point cloud, colors, and estimated intrinsics remain valid.
- The application remains responsive.
- No unbounded CPU or GPU memory growth occurs.

### Pause and Resume

1. Run at least 20 frames.
2. Pause for 10 seconds.
3. Resume.

Pass criteria:

- Pause does not unload the model.
- Temporal caches are preserved.
- Resume continues the same sequence without stale queued frames.
- VRAM remains bounded.

### Stop and Restart

1. Run at least 20 frames.
2. Press Stop.
3. Press Start again on the same source.

Pass criteria:

- STOP invokes the generic adapter sequence-reset contract.
- The restarted run behaves as a fresh sequence.
- No depth result from the old run appears.
- Model weights remain loaded unless explicitly switched or unloaded.

### Video restart and frame rewind

Restart or reopen the video so frame IDs begin again.

Pass criteria:

- The adapter detects the frame counter restart.
- Temporal caches reset before processing the new sequence.
- No geometry is blended with the previous playback.

## 8. Video Depth Anything — USB camera

Run the USB camera for at least five minutes.

Verify:

- Streaming inference remains responsive.
- Pause/Resume preserves temporal state.
- Stop/Start resets temporal state.
- Camera resolution changes reset temporal state.
- Camera disconnect/reconnect creates a fresh session.
- VRAM reaches a stable plateau.
- No cumulative latency increase appears over time.

## 9. Video Depth Anything — RTSP

Run a valid RTSP source for at least five minutes.

Verify:

- Initial connection and inference succeed.
- Temporal output remains stable across normal network jitter.
- Disconnecting and reconnecting creates a fresh sequence.
- A network interruption does not reuse stale temporal caches after recovery.
- Switching RTSP URL or resolution resets the sequence.
- Worker timeout and source recovery behavior remain compatible with Phase 5.

## 10. Output semantics and geometry

### Depth Anything V2

Verify:

- `is_metric = false`
- Depth scale is `relative`
- Intrinsics source is `estimated`
- No confidence map is claimed
- Point-cloud scale is arbitrary

### Video Depth Anything

Verify:

- Metric model checkpoint is selected.
- `is_metric = true`
- Depth scale is `metric`
- Intrinsics source is `estimated`
- No calibrated measurement claim is made
- Any future dimensions would be labelled `Estimated — Not Calibrated`

For both models verify:

- Invalid/non-finite depths do not create invalid points.
- Point colors remain aligned with geometry.
- Portrait and landscape point clouds preserve orientation.
- Min/max depth controls and background removal remain functional.
- Render modes that require unavailable normals/confidence do not falsely claim those outputs.

## 11. Model switching matrix

With one source connected, test all relevant transitions:

- MoGe-2 ↔ Depth Anything V2
- Depth Pro ↔ Depth Anything V2
- UniDepthV2 ↔ Depth Anything V2
- Metric3D v2 ↔ Depth Anything V2
- Depth Anything V2 ↔ Video Depth Anything
- Video Depth Anything ↔ Mock
- Video Depth Anything ↔ every existing real model

Then repeat this complete cycle ten times:

```text
MoGe-2 → Depth Pro → UniDepthV2 → Metric3D v2
→ Depth Anything V2 → Video Depth Anything → Mock → MoGe-2
```

Verify after every switch:

- Source remains connected.
- Old model results do not appear.
- Header badge and footer model/precision update.
- Header total VRAM and footer model VRAM update independently.
- VDA temporal caches are released when switching away.
- VRAM returns near the same baseline for each model.
- No cumulative VRAM increase, OOM, worker loop, or UI freeze occurs.

## 12. RTX 2060 validation

Expected automatic variants:

- Depth Anything V2 Small, ViT-S, FP16, 518 input
- Metric Video Depth Anything Small, ViT-S, FP16, 392 streaming input

Verify:

- Both load without changing configuration manually.
- VDA does not exceed the practical 6 GB device limit during a sustained run.
- The official 518-pixel VDA benchmark reports memory above 6 GB, so the reduced 392-pixel profile must be validated with real sustained inference before acceptance.
- OOM recovery remains available if another GPU application consumes memory.
- Closing other GPU workloads and restarting the worker restores operation.

## 13. RTX 5090 validation

Expected automatic variants:

- Depth Anything V2 Large, ViT-L, FP16, 700 input
- Metric Video Depth Anything Large, ViT-L, FP16, 518 streaming input
- TF32 enabled only when supported and allowed by the hardware profile

Verify:

- Automatic selection needs no code or configuration changes.
- High point budgets remain responsive.
- Long-running VDA caches remain bounded.
- No fallback to the RTX 2060 profile occurs when sufficient memory is available.

## 14. Failure and recovery scenarios

Test or simulate where practical:

### Missing source checkout

Temporarily rename one source directory and load its model.

Expected:

- Clear dependency/source error
- No worker crash loop
- Existing source remains recoverable
- Other models still load

### Missing checkpoint

Temporarily rename one checkpoint and load its model.

Expected:

- Clear download instruction
- No silent remote download in the worker
- No stale model badge or result

### Corrupted checkpoint

Use an invalid test checkpoint.

Expected:

- Controlled load failure
- Model resources cleaned
- Another model can be selected afterward

### CUDA OOM

Create GPU pressure or use a deliberately unsafe override only in a disposable test run.

Expected:

- Failure classified as OOM
- Worker/model cleanup follows existing recovery policy
- VDA caches are cleared
- No endless restart loop
- Efficient profile can be retried

### Worker crash or timeout

Use the existing stress/recovery mechanisms.

Expected:

- Existing Phase 5 single-recovery policy remains intact
- VDA cache state is not reused by the replacement worker
- Source remains connected when the failure is recoverable

### Invalid frame

Verify malformed image shape or output produces a controlled error rather than a native crash.

## 15. Long-running stability

Perform:

- Depth Anything V2 video or RTSP: at least 10 minutes
- Video Depth Anything video or RTSP: at least 10 minutes
- Repeated source and model switching during the run

Monitor:

```bash
watch -n 0.5 nvidia-smi
```

Pass criteria:

- VRAM remains bounded.
- VDA cache growth reaches a stable plateau.
- CPU memory does not grow continuously.
- Latency does not continuously degrade.
- UI remains responsive.
- No worker restart, stale result, or segmentation fault occurs.

## 16. Shutdown validation

Close the application while:

- Idle
- A model is loaded
- Image inference is complete
- Video inference is running
- Camera inference is running
- RTSP inference is running
- Paused
- After a controlled load failure
- After an OOM/recovery test

Then verify:

```bash
pgrep -af "depth_3d_model_lab|core.inference.worker"
```

Pass criteria:

- Application exits with code 0 during normal scenarios.
- No inference worker remains.
- No source thread or child process remains.
- VDA temporal caches and CUDA memory are released.

## 17. Final acceptance

Step 5 is accepted only when:

- Compile validation passes.
- Quick gate passes.
- Full regression gate passes.
- Depth Anything V2 works for all four source types.
- Video Depth Anything works for all four source types.
- Temporal cache preservation/reset behavior matches lifecycle rules.
- RTX 2060 sustained VDA memory is safe and bounded.
- Existing six model/runtime paths remain backward-compatible.
- Manual switching, recovery, stability, and shutdown checks pass.
