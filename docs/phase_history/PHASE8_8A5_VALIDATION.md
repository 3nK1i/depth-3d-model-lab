# Phase 8.8A5 — Controller Hardening / Readiness Gate

## Scope

A5 closes the Phase 8.8A controller layer before any Reconstruction page is enabled. It adds no reconstruction algorithm, no new persistence schema, and no heavy backend dependency to the production `depth` environment.

The hardening work is application-side only:

- live/offline Reconstruction remain mutually exclusive;
- MainWindow supplies resource guards for Benchmark, Snapshot, Stable Video, Measure & Inspect, and inspection export;
- live Reconstruction may continue sharing the existing capture/inference path;
- offline Reconstruction requires inference to be idle and releases an idle resident depth model before the isolated PyCOLMAP job;
- offline session IDs are validated synchronously;
- output directories are rechecked immediately before process launch to close stale-result races;
- a worker result is accepted only when its session identity matches the active controller session;
- failed/cancelled/mismatched results are not forwarded into persistence state;
- a previous process completion must be fully finalized before another offline run starts;
- cancellation retains the existing bounded process-group kill fallback;
- scene/checkpoint mutation is rejected while reconstruction is active;
- shutdown is idempotent and blocks later starts;
- Reconstruction navigation remains disabled and the main stack remains unchanged.

## Focused automated validation

```bash
./scripts/run_phase8_8a5_tests.sh quick
./scripts/run_phase8_8a5_tests.sh integration
./scripts/run_phase8_8a5_tests.sh backward
```

Expected A5 delivery counts are `11 passed` for quick and `2 passed` for integration. The backward gate must preserve A4 (`15 passed`, `1 passed`) and its A3 backward groups (`20 passed`, `1 passed`).

The A5 integration tests explicitly cover:

- wrong-session result rejected although the worker exits 0 and claims pass;
- worker crash does not alter persistence state;
- cooperative cancellation cannot publish a result even if the worker writes one during SIGTERM handling;
- a clean job can start and complete after failure/cancellation;
- a SIGTERM-resistant worker and child process are removed by the existing process-group escalation path.

## Required real/controller validation before commit

A5 changes the controller entry points and offline result-admission boundary, so re-run the already-qualified real live and offline paths rather than inventing another backend test.

### Real Stella controller path

```bash
rm -rf /tmp/phase8_8a2_live_controller_stella
python scripts/run_phase8_8a2_live_controller_stella_smoke.py \
  --max-frames 500 \
  --playback-speed 2.0
```

Expected: `Phase 8.8A2 ReconstructionController live stella smoke: PASS`.

### Real offline PyCOLMAP controller path

```bash
rm -rf /tmp/phase8_8a3_offline_controller_pycolmap
python scripts/run_phase8_8a3_offline_controller_pycolmap_smoke.py \
  --images /home/eiatech/Desktop/phase8_3c_south_building/south-building/images \
  --sfm-python ~/anaconda3/envs/eia-pycolmap-probe/bin/python \
  --max-keyframes 50
```

Expected: `Phase 8.8A3 ReconstructionController offline PyCOLMAP smoke: PASS`, with `tests_fixture_used=false`, external PyCOLMAP isolation, and an accepted SfM quality assessment.

### Scene/checkpoint/resume bridge

```bash
rm -rf /tmp/phase8_8a4_scene_checkpoint_bridge
python scripts/run_phase8_8a4_scene_checkpoint_bridge_smoke.py \
  --offline-result /tmp/phase8_8a3_offline_controller_pycolmap/offline_result.json
```

Expected: `Phase 8.8A4 scene/checkpoint/resume bridge smoke: PASS`.

### Tiny application sanity

```bash
python main.py
```

Verify only the affected boundary: application starts/exits cleanly, existing modes remain normal, Reconstruction remains disabled, and no Reconstruction page is visible.

## Commit gate

Do not commit until focused tests, integration hardening, backward gates, `git diff --check`, the real Stella/PyCOLMAP controller smokes, the A4 persistence smoke, and the tiny application sanity are all green.
