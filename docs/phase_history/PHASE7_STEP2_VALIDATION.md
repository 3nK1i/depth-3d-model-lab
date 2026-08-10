# Phase 7 Step 2 — Focused Manual Validation

Only Snapshot Mode and directly affected compatibility paths need manual validation.

1. Load an image in Live 3D, connect it, select Development Mock, and open Snapshot.
2. Capture the latest frame; confirm the frozen RGB frame and resolution appear.
3. Run Snapshot; confirm the UI remains responsive and RGB, depth, and 3D results appear.
4. Open the timestamped export and confirm `rgb.png`, `depth.npy`, `depth_preview.png`, `point_cloud.ply`, `snapshot_manifest.json`, and `result.npz` exist.
5. Repeat with one real model. Confirm metric/relative and intrinsics metadata in the manifest are correct.
6. Start a slow Marigold snapshot and cancel it. Confirm no final incomplete export directory is presented and a new run can start.
7. Capture from one continuous source (Video, USB, or RTSP) and confirm the captured frame remains frozen while the source continues.
8. Minimal compatibility: return to Live 3D and run one normal inference; open Benchmark and run the Development Mock smoke benchmark.
