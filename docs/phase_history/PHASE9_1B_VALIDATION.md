# Phase 9.1B Validation — Calibration Resolution Compatibility

## Scope

Phase 9.1B adds an explicit application layer for using an immutable camera
calibration profile on a target image grid.

Qualified paths in this step:

- exact calibration resolution;
- explicitly requested, same-aspect-ratio full-frame resize.

Rejected by design:

- implicit resolution changes;
- aspect-ratio changes;
- crop or padding inference;
- digital zoom or stabilisation crop inference;
- changed field-of-view/sensor-mode assumptions.

The implementation reuses the existing `ImageTransform2D`,
`declare_full_frame_resize()` and `transform_intrinsics()` contracts. Existing
live reconstruction resolution guards remain unchanged.

## Authority guarantees

- The original calibration profile remains immutable at its calibration
  resolution.
- `calibration_record_id`, calibrated status and distortion coefficients are
  preserved when a qualified full-frame resize is applied.
- Resolution adaptation does not create metric scale or certified metrology.
- No crop, sensor mode, zoom or field-of-view change is inferred from image
  dimensions alone.

## Focused validation

```bash
./scripts/run_phase9_1b_tests.sh quick
./scripts/run_phase9_1b_tests.sh integration
python scripts/run_phase9_1b_calibration_smoke.py
```

The integration gate verifies that a safely rescaled camera satisfies the
existing `LiveSlamRequest` exact source/camera resolution contract without
weakening or bypassing that Phase 8 safety check.
