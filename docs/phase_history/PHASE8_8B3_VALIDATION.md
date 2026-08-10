# Phase 8.8B3 — Reconstruction Runtime Controls

## Scope

B3 activates the dormant Reconstruction page's runtime controls over the already-hardened Phase 8.8A controller layer. It adds no reconstruction algorithm, no new native dependency, and does not enable Reconstruction navigation.

### Runtime behavior

- A ready **Image sequence** or **Recorded video** enables **Start** and launches the existing isolated A3 offline worker through `ReconstructionController.start_selected_offline()`.
- A ready **Live source** enables **Start** and then requires an explicit camera-geometry choice:
  - **Use calibration file** loads the existing calibrated-camera JSON path.
  - **Run Test Mode** uses the existing Phase 8.6 estimated-intrinsics contract with an assumed 70° horizontal FOV, centered principal point, square pixels, zero distortion, and explicit focal-length uncertainty.
- Test Mode is opt-in and remains `ESTIMATED / NOT_CALIBRATED`, uses `CameraIntrinsicsPolicy.ALLOW_ESTIMATED`, keeps monocular scale arbitrary, and never enables certified metrology.
- **Pause / Resume** is live-only and delegates to the A2 live bridge.
- **Stop** is non-blocking for live reconstruction; **Cancel** is used for offline reconstruction.
- Source controls are disabled while a reconstruction job owns resources and re-enable when it reaches a terminal state.
- Offline runtime output is transient under `/tmp/eia_depth_reconstruction/<session-id>` by default. Phase 8.8G owns final session/output UX.

### Qualified external runtime defaults

The production `depth` environment still imports none of the heavy reconstruction backends. B3 resolves the already-qualified external paths at Start time:

- PyCOLMAP Python: `~/anaconda3/envs/eia-pycolmap-probe/bin/python`
- Stella prefix: `~/Desktop/eia-stella-vslam-probe`
- Stella worker: `~/Desktop/eia-stella-live-worker/bin/eia_stella_live_worker`
- Stella ORB vocabulary: `~/Desktop/phase8_4_stella_probe_data/orb_vocab.fbow`

Environment overrides are supported with `EIA_RECONSTRUCTION_PYCOLMAP_PYTHON`, `EIA_RECONSTRUCTION_OUTPUT_ROOT`, `EIA_RECONSTRUCTION_STELLA_PREFIX`, `EIA_RECONSTRUCTION_STELLA_WORKER`, and `EIA_RECONSTRUCTION_STELLA_VOCAB`.

## Validation

```bash
./scripts/run_phase8_8b3_tests.sh quick
./scripts/run_phase8_8b3_tests.sh backward
```

Real offline UI/controller gate:

```bash
rm -rf /tmp/phase8_8b3_offline_runtime_ui
python scripts/run_phase8_8b3_offline_runtime_ui_smoke.py \
  --images /home/eiatech/Desktop/phase8_3c_south_building/south-building/images \
  --sfm-python ~/anaconda3/envs/eia-pycolmap-probe/bin/python
```

Expected: `Phase 8.8B3 offline runtime UI smoke: PASS`.

Because B3 does not change the low-level live runtime, retain the real A2 Stella smoke as the live-backend backward gate.

The B3 pre-commit calibration correction extends that existing live request boundary with the
already-qualified `CameraIntrinsicsPolicy` field. Calibrated starts retain the historical
`CALIBRATED_ONLY` default; Test Mode explicitly opts into `ALLOW_ESTIMATED` rather than silently
guessing calibration.

Real no-calibration/Test-Mode gate:

```bash
rm -rf /tmp/phase8_8b3_live_test_mode_stella
python scripts/run_phase8_8b3_live_test_mode_stella_smoke.py \
  --max-frames 500 \
  --playback-speed 2.0
```

Expected: `Phase 8.8B3 live Test Mode Stella smoke: PASS`, with estimated/not-calibrated
intrinsics, arbitrary/non-metric authority, and the existing isolated Stella worker.

## Locked invariants

- Reconstruction remains outside `_ENABLED_MODES`.
- `MainWindow._on_mode_changed()` still has no Reconstruction branch.
- Source selection alone never launches a backend.
- Live Test Mode is always explicit; estimated intrinsics are never labelled calibrated.
- Estimated/Test live reconstruction remains non-metric and non-certified.
- Live capture ownership remains `ApplicationController`.
- Offline PyCOLMAP remains an external process.
- Sparse A3 output remains non-canonical and non-certified.
- Persistent-scene/checkpoint authority remains unchanged.
