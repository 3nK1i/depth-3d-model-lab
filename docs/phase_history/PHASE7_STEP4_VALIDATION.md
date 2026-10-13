# Phase 7 Step 4 — Focused Release Validation

## 1. Compile and automated tests

```bash
python -m compileall -q app core tests scripts main.py
./scripts/run_phase7_step4_tests.sh quick
./scripts/run_phase7_step4_tests.sh full
```

## 2. Refresh the benchmark report

The version is now `0.7.0`, so rerun the registry benchmark to generate a matching report:

```bash
python scripts/run_phase6_benchmark.py \
  --models all \
  --profile auto \
  --point-budget 100000 \
  --require-all
```

## 3. Focused manual release scenarios

Only release-level interactions introduced by Phase 7 need manual confirmation.

### Cross-mode navigation

- Open Live 3D, Stable Video, Snapshot and Benchmark repeatedly.
- Confirm each page remains responsive and preserves its latest completed result.
- Confirm Reconstruction remains disabled.

### Snapshot artifact

- Use one real model and one image.
- Stop Live 3D inference, capture the frame and run Snapshot.
- Confirm depth and point cloud display correctly.
- Confirm `rgb.png`, `depth.npy`, `depth_preview.png`, `point_cloud.ply` and `snapshot_manifest.json` exist.
- Confirm the manifest says metric or relative correctly and never claims calibrated intrinsics.

### Stable Video artifact

- Process at least 30 frames from a Video, Camera or RTSP source.
- Pause and resume once, then Stop & Finalize.
- Confirm `rgb_video.mp4` and `depth_video.mp4` are playable and synchronized.
- Confirm `performance.csv` row count matches the manifest frame count.
- Confirm no partial export directory remains.

### Benchmark sanity

- Open Benchmark and confirm the refreshed report loads.
- Run a Development Mock smoke benchmark from the UI.

### Shutdown

- Close the application after visiting all four operational modes.
- Verify no worker remains:

```bash
pgrep -af "depth_3d_model_lab|core.inference.worker"
```

## 4. Release gate before commit

```bash
./scripts/run_phase7_release.sh full \
  --require-benchmark \
  --require-snapshot-export \
  --require-stable-video-export
```

A dirty-worktree warning is expected before the release commit. Every required check must pass.

## 5. Commit, clean-tree gate, tag and push

```bash
git add -A
git commit -m "release: finalize phase 7 operational modes"

./scripts/run_phase7_release.sh full \
  --require-benchmark \
  --require-snapshot-export \
  --require-stable-video-export \
  --require-clean-git

git tag -a v0.7.0 \
  -m "Depth & 3D Model Lab v0.7.0"

git push origin HEAD --follow-tags
```
