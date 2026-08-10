# Phase 7 Step 3 — Stable Video Mode Validation

## Scope

This checklist validates only Stable Video Mode and directly affected runtime behavior. The complete application regression remains covered by the automated suite.

## Automated validation

```bash
python -m compileall -q app core tests scripts main.py
./scripts/run_phase7_step3_tests.sh quick
./scripts/run_phase7_step3_tests.sh full
```

## Focused manual validation

### 1. Mode and source eligibility

- Confirm **Stable Video** is enabled and **Reconstruction** remains disabled.
- With no source connected, Start must remain unavailable.
- A static Image must be rejected with guidance to use Snapshot Mode.
- Connect a Video, USB Camera, or RTSP source in Live 3D; Stable Video must become available.

### 2. Stateless model stabilization

Use MoGe-2, Depth Pro, UniDepthV2, Metric3D v2, or Depth Anything V2.

- Run the same video with stabilization **Off** and then **Medium**.
- Confirm depth-map and point-cloud flicker decreases with Medium.
- Confirm **High** is steadier but may introduce additional temporal lag.
- Confirm metric/relative depth semantics remain unchanged.

### 3. Native temporal model

Use Video Depth Anything.

- Confirm the page reports `Native model cache + optional smoothing`.
- Start, Pause for several seconds, then Resume.
- Confirm temporal state continues without restarting.
- Stop & Finalize, then start a new run.
- Confirm the previous sequence does not contaminate the new run.

### 4. Backpressure and statistics

Use a 25–30 FPS source with a slower model.

- The UI must remain responsive.
- Source FPS may exceed Processing FPS.
- Dropped Frames should rise instead of latency growing without bound.
- RGB and depth displays must remain synchronized.
- Average latency, peak latency, resets, Model VRAM, and elapsed time must update.

### 5. Sequence reset

- Let a looping video restart or reconnect an RTSP source.
- Confirm a temporal-reset message appears when frame IDs restart.
- Change source resolution if supported and confirm a reset is recorded.
- Processing must continue without mixing the previous temporal sequence.

### 6. Export

Run at least 30 processed frames and click **Stop & Finalize**.

```bash
find outputs/stable_video -maxdepth 2 -type f | sort
```

Verify:

- `rgb_video.mp4` is playable.
- `depth_video.mp4` is playable and synchronized.
- `performance.csv` contains per-frame timing and drop counters.
- `run_manifest.json` records model, temporal mode, smoothing, depth scale, intrinsics source, frame count, and reset count.
- When Numerical Depth is enabled, `depth_frames/*.npy` exists.
- **Open Export** opens the completed directory.

### 7. Minimal compatibility

- Return to Live 3D and run one image inference.
- Confirm the previous Live 3D temporal-smoothing value is restored after Stable Video stops.
