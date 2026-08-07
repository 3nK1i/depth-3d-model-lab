# Phase 8.4D Validation — Live SLAM Hardening & Finalization

Phase 8.4D hardens the live SLAM stack completed in 8.4A–8.4C without enabling Reconstruction UI, loop closure/global optimization, or dense fusion.

## Scope

- bounded external worker restart after crash, timeout, or malformed IPC
- explicit `RECOVERING` runtime state
- bounded stale-frame tolerance
- latest-frame-only behavior preserved during worker recovery
- tracking-loss/recovery counters
- latency / replacement / restart / cleanup telemetry
- deterministic stop escalation and resource cleanup
- process-group termination for the isolated native worker
- source reconnect pause/resume behavior preserved from 8.4C
- repeated start/stop shared-memory leak checks
- real stella crash/restart smoke through production `VideoSource`
- strict arbitrary-scale pose semantics preserved
- Reconstruction UI remains disabled

## Intentionally deferred

- loop closure / relocalization / pose-graph optimization: Phase 8.5
- dense depth fusion: Phase 8.6
- mesh/texture/semantic refinement: Phase 8.7
- Reconstruction UI enablement: Phase 8.8

## Focused tests

```bash
./scripts/run_phase8_4d_tests.sh quick
./scripts/run_phase8_4d_tests.sh integration
./scripts/run_phase8_4d_tests.sh stress
python scripts/run_live_slam_hardening_smoke.py
```

Expected focused results for this package:

- quick: 7 passed
- integration: 3 passed
- stress: 2 passed
- hardening smoke: PASS

## Real isolated stella recovery gate

The qualified native worker from Phase 8.4B must already exist at:

```text
~/Desktop/eia-stella-live-worker/bin/eia_stella_live_worker
```

Run:

```bash
rm -rf /tmp/phase8_4d_live_capture_stella_recovery

python scripts/run_live_capture_stella_recovery_smoke.py \
  --max-frames 800 \
  --crash-after 250 \
  --playback-speed 2.0
```

This deliberately kills the first isolated stella worker once. The live capture runtime must restart only the worker and continue using the existing capture owner and bounded shared-memory ring.

Required summary properties:

```text
pass=true
worker_restarts=1
worker_restart_failures=0
recovery_state_observed=true
cleanup_complete=true
backend_resources_clean=true
leaked_worker_sockets=[]
leaked_shared_memory=[]
real_stella_executed=true
worker_process_isolated=true
capture_source=VideoSource
capture_owner=existing_capture_layer
capture_thread_blocking=false
latest_frame_only=true
pose_convention=T_reconstruction_from_camera
scale_status=arbitrary
unit_to_metre=null
loop_closure_enabled=false
global_optimization_feature_enabled=false
```

The report is written to:

```text
/tmp/phase8_4d_live_capture_stella_recovery/summary.json
```

## Backward gate

```bash
./scripts/run_phase8_4c_tests.sh quick
./scripts/run_phase8_4b_tests.sh quick
./scripts/run_phase8_4a_tests.sh quick
./scripts/run_phase8_3d_tests.sh quick
./scripts/run_phase8_3c_tests.sh quick
./scripts/run_phase8_3b_tests.sh quick
./scripts/run_phase8_2_tests.sh quick
./scripts/run_phase8_1_tests.sh quick
```

## Isolation gate

```bash
grep -RInE \
'^[[:space:]]*(from[[:space:]]+pycolmap|import[[:space:]]+pycolmap)' \
core/reconstruction \
&& echo "ERROR: direct PyCOLMAP import found" \
|| echo "PASS: core/reconstruction remains PyCOLMAP-free"
```

```bash
grep -RInE \
'^[[:space:]]*(from[[:space:]]+stella_vslam|import[[:space:]]+stella_vslam)|ctypes\.(CDLL|PyDLL)' \
core/reconstruction \
&& echo "ERROR: native stella linkage found in Python" \
|| echo "PASS: stella_vslam remains external-process isolated"
```

## Full regression

```bash
./scripts/run_phase8_4d_tests.sh full
python main.py
```

Manual sanity is intentionally small: confirm the app starts/exits normally, existing modes remain unchanged, and Reconstruction remains disabled.

Do not create a release tag in Phase 8.4D. `v0.8.0` remains reserved for Phase 8.9.
