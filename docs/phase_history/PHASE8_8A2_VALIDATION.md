# Phase 8.8A2 Validation — Live Reconstruction Lifecycle Bridge

Baseline: `phase8-reconstruction` commit `7885383379e2ee6311425f5b32c5dc9bebc01c48`.

## Scope

Phase 8.8A2 makes the Phase 8.8 `ReconstructionController` the public
application-facing lifecycle bridge for the already-qualified live SLAM stack.
It does **not** enable Reconstruction in the UI and does not add any offline
reconstruction execution path.

Implemented in this step:

- `ReconstructionController.start_live(...)` delegates to the existing
  `LiveSlamCaptureController.start_stella_for_active_source(...)`;
- live session identity is retained by the application controller boundary;
- pause and resume remain thin delegates to the authoritative
  `LiveCaptureSlamRuntime` state machine;
- `request_stop_live()` provides a cooperative non-blocking stop path suitable
  for a future UI button;
- `stop_live()` remains a bounded synchronous teardown for controlled
  test/shutdown boundaries;
- current live runtime statistics are exposed without translating them;
- source reconnect/error/type-change behavior remains owned by the existing
  Phase 8.4 live bridge;
- `ApplicationController.frame_captured` remains the single capture feed;
- focused mock integration validates the complete
  `ApplicationController -> ReconstructionController -> LiveSlamCaptureController
  -> LiveCaptureSlamRuntime` path;
- a real controller-level EuRoC/VideoSource/stella smoke validates the isolated
  native worker through the new Phase 8.8A2 application boundary.

## Explicitly unchanged / deferred

- Reconstruction navigation remains disabled.
- No Reconstruction page is added to the main stack.
- `MainWindow` does not auto-start live reconstruction.
- No camera/RTSP/video capture subsystem is duplicated.
- No direct native stella linkage is added to Python.
- PyCOLMAP, Gaussian, NeRF and DA3 remain externally isolated.
- Offline image/video reconstruction execution remains Phase 8.8A3.
- UI controls remain Phase 8.8B.
- Live 3D visualization remains Phase 8.8C.

## Focused automated validation

```bash
./scripts/run_phase8_8a2_tests.sh quick
./scripts/run_phase8_8a2_tests.sh integration
./scripts/run_phase8_8a2_tests.sh backward
```

The backward gate intentionally re-runs the mature Phase 8.4D live hardening
quick suite and the Phase 8.8A1 controller/UI lock suite.

## Real isolated stella gate — required before commit

The already-qualified Phase 8.4 native worker/data are expected at their
existing locations. Run:

```bash
rm -rf /tmp/phase8_8a2_live_controller_stella

python scripts/run_phase8_8a2_live_controller_stella_smoke.py \
  --max-frames 500 \
  --playback-speed 2.0

cat /tmp/phase8_8a2_live_controller_stella/summary.json
```

Required high-level invariants:

- `pass: true`
- `controller: "ReconstructionController"`
- `live_bridge: "LiveSlamCaptureController"`
- `capture_owner: "ApplicationController"`
- `controller_path_exercised: true`
- a real session id is retained
- STARTING, RUNNING, PAUSED, STOPPING and STOPPED are observed
- pause/resume succeeds
- capture continues while reconstruction is paused
- the non-blocking stop request succeeds
- final cleanup succeeds
- real stella observations/poses are received
- worker remains external-process isolated
- latest-frame-only semantics remain active
- pose convention remains `T_reconstruction_from_camera`
- monocular scale remains arbitrary with `unit_to_metre: null`

## Static isolation / UI lock

```bash
git diff --check

python -m compileall -q \
  app/reconstruction_controller.py \
  app/live_slam_controller.py \
  tests/unit/test_phase8_8a2_reconstruction_controller.py \
  tests/integration/test_phase8_8a2_live_controller_pipeline.py \
  scripts/run_phase8_8a2_live_controller_stella_smoke.py

grep -RInE \
'^[[:space:]]*(from[[:space:]]+pycolmap|import[[:space:]]+pycolmap|from[[:space:]]+stella_vslam|import[[:space:]]+stella_vslam)|ctypes\.(CDLL|PyDLL)' \
core/reconstruction app/reconstruction_controller.py app/live_slam_controller.py \
&& echo "ERROR: forbidden direct native import/linkage found" \
|| echo "PASS: heavy/native reconstruction boundaries remain isolated"
```

The existing Phase 8.8A1 UI test must continue to prove that Reconstruction is
disabled and the main page stack still contains only the five existing pages.

Do not commit Phase 8.8A2 until the automated gates, real controller-level
stella smoke, `git diff --check`, and a tiny application start/exit sanity are
all clean. Do not create a release tag.
