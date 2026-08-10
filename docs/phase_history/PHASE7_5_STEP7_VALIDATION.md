# Phase 7.5 Step 7 Validation

## Scope

Step 7 adds dependency-light persistent object tracking and temporal measurement stabilization. It does not activate tracking in the application UI or modify the existing depth/perception workers.

## Architecture

- Same-class deterministic association using dense-mask IoU, detection-box IoU, and normalized centre distance.
- Persistent track IDs with tentative, confirmed, lost, reacquired, and terminated states.
- Bounded per-kind measurement histories.
- Robust temporal median and MAD outlier rejection.
- Combined per-frame and temporal uncertainty components.
- Circular stabilization for roll, pitch, and yaw.
- Reset on source session, resolution, depth model, perception model, worker generation, or alignment configuration changes.

## Files

### Added

- `core/tracking/association.py`
- `core/tracking/factories.py`
- `core/tracking/pipeline.py`
- `core/tracking/policy.py`
- `core/tracking/stabilization.py`
- `core/tracking/temporal_contracts.py`
- `scripts/run_temporal_tracking_smoke.py`
- `scripts/run_phase7_5_step7_tests.sh`
- `tests/tracking_helpers.py`
- Step 7 unit, integration, and stress tests

### Replaced

- `core/tracking/contracts.py`
- `core/tracking/validation.py`
- `core/tracking/__init__.py`

The replacement preserves the existing `TrackingState`, `TrackingResetReason`, and `ObjectTrackState` public semantics. New measurement-dependent exports are loaded lazily to prevent a circular dependency with `core.measurement.contracts`.

### Removed

None.

## Validation commands

```bash
python -m compileall -q app core scripts tests
./scripts/run_phase7_5_step7_tests.sh quick
./scripts/run_phase7_5_step7_tests.sh integration
./scripts/run_phase7_5_step7_tests.sh full
```

Smoke visualization:

```bash
python scripts/run_temporal_tracking_smoke.py \
  --output /tmp/phase7_5_step7_tracking.png
```

## Focused manual validation

1. Launch the application.
2. Confirm Live 3D, Stable Video, Snapshot, and Benchmark remain unchanged.
3. Confirm Reconstruction remains disabled.
4. Confirm no tracking or stabilized-measurement controls appear.
5. Run Development Mock Live 3D.
6. Run one Snapshot export.
7. Close the application and verify no worker remains.
8. Run and inspect the standalone Step 7 smoke visualization.

## Limitations

- Association is geometry-based; appearance embeddings are deferred.
- The engine is not wired into the production application runtime in Step 7.
- Temporal intervals remain estimated and are not empirically calibrated coverage intervals.
- Full UI integration belongs to Step 8.

## Git

After every applicable automated and manual gate passes:

```bash
git status --short
git add -A
git commit -m "feat(tracking): add Phase 7.5 temporal stabilization"
git push
```

No version bump or tag is required.

## Delivery-environment results

Executed in the delivery container:

- Compile validation: passed
- Step 7 quick gate: 21 passed
- Step 7 integration gate: 1 passed
- Step 7 focused stress sanity: 3 passed
- Step 1 quick regression: 39 passed
- Step 2 quick regression: 34 passed
- Step 3 quick regression: 39 passed
- Step 4 quick regression: 32 passed
- Step 5 quick regression: 21 passed
- Step 6 quick regression: 20 passed
- Critical depth/model/IPC/measurement contracts: 56 passed
- Synthetic tracking smoke: passed with one persistent ID, injected-outlier rejection, loss, and reacquisition

The complete shared-process unit/UI/integration/stress suite could not be claimed in this container because PyQt6 and VisPy are not installed. The full split-process gate must be run in the configured Python 3.11 `depth` environment.
