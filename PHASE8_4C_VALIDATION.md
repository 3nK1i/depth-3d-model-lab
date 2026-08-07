# Phase 8.4C — Live Capture Integration Validation

Starting point: `phase8-reconstruction` at `019447d53aced7f4b06530e0d71ea20e288cfc90`.

## Scope

Phase 8.4C connects the existing capture layer to the already-qualified isolated stella_vslam worker without changing capture ownership or enabling the Reconstruction UI.

Implemented in this step:

- dormant `LiveSlamCaptureController` attached to `ApplicationController.frame_captured`
- USB, RTSP, and recorded-video source identity/metadata support
- explicit calibrated-camera JSON loader/exporter; no estimated intrinsics are called calibrated
- non-blocking latest-frame-only live runtime on a dedicated Python thread
- bounded shared-memory publication to the Phase 8.4B backend
- pause/resume/stop handling for capture reconnect/disconnect lifecycle
- real recorded-video -> existing `VideoSource` -> `SourceFrame` -> isolated stella worker validation script
- focused unit/integration/stress coverage and backward-compatibility checks

Deferred by design:

- loop closure / relocalization / global optimization: Phase 8.5
- dense fusion: Phase 8.6
- Reconstruction UI enablement: Phase 8.8

## Apply

```bash
cd /home/eiatech/Desktop/depth_3d_model_lab
conda activate depth
unzip -o ~/Downloads/eia_depth_phase8_4c_live_capture_integration.zip -d .
chmod +x scripts/run_phase8_4c_tests.sh
chmod +x scripts/run_live_capture_runtime_smoke.py
chmod +x scripts/run_live_capture_stella_smoke.py
git diff --check
```

## Focused validation

```bash
./scripts/run_phase8_4c_tests.sh quick
./scripts/run_phase8_4c_tests.sh integration
./scripts/run_phase8_4c_tests.sh stress
python scripts/run_live_capture_runtime_smoke.py
```

The quick suite may include PyQt-backed source identity tests on the target `depth` environment.

## Real recorded-video / stella gate

Uses the already-qualified Phase 8.4B native worker and EuRoC MH_01_easy data. It creates a temporary calibrated MJPEG video and plays it through the production `VideoSource` QThread path.

```bash
rm -rf /tmp/phase8_4c_live_capture_stella
python scripts/run_live_capture_stella_smoke.py --max-frames 500
cat /tmp/phase8_4c_live_capture_stella/summary.json
```

Required summary invariants include:

- `pass: true`
- `real_stella_executed: true`
- `worker_process_isolated: true`
- `capture_source: "VideoSource"`
- `capture_owner: "existing_capture_layer"`
- `capture_thread_blocking: false`
- `latest_frame_only: true`
- `pose_convention: "T_reconstruction_from_camera"`
- `scale_status: "arbitrary"`
- `unit_to_metre: null`
- loop closure/global optimization false

## Backward and isolation gates

```bash
./scripts/run_phase8_4b_tests.sh quick
./scripts/run_phase8_4a_tests.sh quick
./scripts/run_phase8_3d_tests.sh quick
./scripts/run_phase8_3c_tests.sh quick
./scripts/run_phase8_3b_tests.sh quick
./scripts/run_phase8_2_tests.sh quick
./scripts/run_phase8_1_tests.sh quick

grep -RInE '^[[:space:]]*(from[[:space:]]+pycolmap|import[[:space:]]+pycolmap)' core/reconstruction \
  && echo 'ERROR: direct PyCOLMAP import found' \
  || echo 'PASS: core/reconstruction remains PyCOLMAP-free'

grep -RInE '^[[:space:]]*(from[[:space:]]+stella_vslam|import[[:space:]]+stella_vslam)|ctypes\.(CDLL|PyDLL)' core/reconstruction \
  && echo 'ERROR: native stella linkage found in Python' \
  || echo 'PASS: stella_vslam remains external-process isolated'
```

## Full gate

```bash
./scripts/run_phase8_4c_tests.sh full
python main.py
```

Manual sanity is intentionally small: app starts/exits normally, existing modes are unchanged, and Reconstruction remains disabled.

Do not commit until the real capture/stella smoke and full regression pass. Do not create a tag for Phase 8.4C.
