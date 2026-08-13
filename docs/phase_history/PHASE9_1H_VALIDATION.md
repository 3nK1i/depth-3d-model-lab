# Phase 9.1H Validation — Calibration / Metric Persistence & Reproducibility

Phase 9.1H persists Phase 9 camera-calibration and metric-qualification evidence without changing the qualified Phase 8 persistent-scene manifest schema.

## Scope

- Keep `persistent_reconstruction_scene.json` schema v1 and its fingerprint semantics unchanged.
- Add an integrity-checked `persistent_reconstruction_authority.json` sidecar bound to scene ID, scene fingerprint, and session ID.
- Snapshot the full selected `CameraCalibrationProfile`; reload does not depend on the original profile-store file.
- Rejected camera-calibration records cannot be persisted/restored as the active scene calibration.
- Snapshot optional `MetricQualificationEvidence` and its recomputed authority assessment/scale uncertainty.
- Preserve `certified_metrology = false`.
- Treat a v0.8.0/legacy scene with no Phase 9 authority sidecar as having no persisted calibration/metric qualification evidence.
- Restore persisted calibration into the shared Reconstruction / Measure & Inspect selection on scene load.

## Authority rules

- Camera calibration never establishes metric scale by itself.
- Metric qualification is restored only from persisted explicit evidence.
- Rejected/supporting-only/unsupported evidence cannot gain authority during reload.
- Gaussian/NeRF authority is unchanged.

## Validation

Run:

```bash
./scripts/run_phase9_1h_tests.sh quick
./scripts/run_phase9_1h_tests.sh integration
python scripts/run_phase9_1h_persistence_smoke.py
```

Manual validation is limited to loading a Phase 9 scene with a sidecar and one legacy scene without a sidecar, confirming calibration/metric status restores truthfully and the UI remains responsive at 1366×768.
