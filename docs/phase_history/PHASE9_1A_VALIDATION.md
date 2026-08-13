# Phase 9.1A — Calibration Contracts & Compatibility Foundation

## Scope

Phase 9.1A is additive and dependency-light. It introduces a reusable camera-calibration profile contract around the existing Phase 8 `ReconstructionCamera` authority.

It does **not** implement checkerboard/ChArUco solving, resolution rescaling, metric qualification, UI workflow, measurement integration, or persistent-scene migration yet.

## Locked authority rules

- Camera calibration does not establish metric scale.
- Metric scale remains owned by `ScaleReferenceProfile` and explicit scale evidence.
- `certified_metrology` remains false under the existing Phase 8/9 reconstruction authority model.
- Existing Phase 8 live-camera calibration JSON remains readable/exportable without changing its schema.
- Legacy live calibration is marked `legacy_compatible`; it is not silently upgraded to Phase 9 validation evidence.
- Heavy reconstruction runtimes and the production Conda environment are unchanged.

## Added foundation

- `CameraCalibrationProfile`
- `CameraCalibrationMethod`
- `CameraCalibrationValidationState`
- `CalibrationSourceBinding`
- `CalibrationQualityMetrics`
- deterministic JSON save/load helpers
- Phase 8 live-calibration import/export bridge
- source identity/fingerprint binding without persisting source URI/credentials
- explicit calibration-resolution binding when source metadata declares a resolution

## Validation

Quick gate:

```bash
./scripts/run_phase9_1a_tests.sh quick
```

Integration gate:

```bash
./scripts/run_phase9_1a_tests.sh integration
```

Dependency-light smoke:

```bash
python scripts/run_phase9_1a_calibration_smoke.py
```

Phase 9.1B will add explicit resolution-compatibility/application policy using the existing `ImageTransform2D` / `transform_intrinsics()` foundation rather than silently rescaling or misusing intrinsics.
