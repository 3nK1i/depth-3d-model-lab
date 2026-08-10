# Phase 7.5 Step 8 — Final Rendering, Status, and MoGe Metric Fix

## Baseline

This correction is applied after:

1. `eia_depth_phase7_5_step8_complete_files.zip`
2. `eia_depth_phase7_5_step8_ui_release_test_fix_complete_files.zip`
3. `eia_depth_phase7_5_step8_runtime_uncertainty_logging_fix_complete_files.zip`

## Corrected defects

### Qt label painting

- Uses `QRectF` with `QPointF` for scaled label placement.
- Clamps labels to the displayed image area.
- Always ends the `QPainter` through `try/finally`.
- Prevents `QBackingStore::endPaint()` and paint-device shutdown warning cascades caused by a paint exception.

### Status semantics

- `RUNNING` / model stages: cyan.
- `COMPLETE`: green.
- `FAILED`: red.
- `CANCELLING` / `CANCELLED`: amber.
- The progress bar now follows the same state colour as the badge.

### MoGe-2 metric geometry

- Converts the official normalized MoGe intrinsic matrix to full-resolution pixel-space intrinsics before depth back-projection.
- Prevents lateral coordinates and camera distance from being inflated by image-width/image-height scale factors.
- Keeps the application-wide intrinsics contract in pixel coordinates.

### Measurement presentation

- Width, height, and depth remain displayed in millimetres.
- Camera distance and centre coordinates remain in metres with readable precision.

## Validation completed in the delivery environment

- Python compile validation: passed.
- Step 8 quick gate: 16 passed.
- Step 8 integration gate: 1 passed.
- Focused MoGe/inspection unit tests: 10 passed.

The complete Qt/VisPy suite was not run in the delivery environment because PyQt6 and VisPy are not installed there. The supplied regression tests must be run in the configured `depth` environment.

## Target validation

```bash
python -m compileall -q app core scripts tests
./scripts/run_phase7_5_step8_tests.sh quick
./scripts/run_phase7_5_step8_tests.sh integration
./scripts/run_phase7_5_step8_tests.sh ui
./scripts/run_phase7_5_step8_tests.sh full
```

Then run one real Measure & Inspect analysis with:

- MoGe-2
- Ultralytics YOLO Instance Segmentation
- Efficient profile
- 3 objects
- Labels enabled

Expected:

- No `QPointF`/`QPoint` exception.
- No `QBackingStore::endPaint()` warning cascade.
- `COMPLETE` badge and progress bar are green.
- A forced model failure displays a red `FAILED` badge and red progress bar.
- Camera distance and centre coordinates are physically plausible metre-scale values rather than kilometre-scale values.
