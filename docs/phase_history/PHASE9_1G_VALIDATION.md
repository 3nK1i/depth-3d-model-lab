# Phase 9.1G Validation — Application-Wide Calibration Integration

Phase 9.1G connects the shared Phase 9 camera-calibration profile to Measure & Inspect while preserving Reconstruction behavior and the Phase 8/9 authority model.

## Scope

- Reuse the single Reconstruction-owned calibration profile selection in Measure & Inspect.
- Rectify captured measurement frames when the selected calibration contains Brown-Conrady or OpenCV fisheye distortion.
- Run perception and depth on the same rectified image grid.
- Transform trusted calibrated intrinsics into the native depth grid and inject them through an explicit alignment override.
- Preserve the default factory rejection of model-declared calibrated intrinsics that lack a calibration record.
- Preserve the depth model's original metric/relative/unknown scale semantics; camera calibration alone never qualifies metric scale.
- Persist calibration profile/record/method/validation/application provenance in temporary inspection bundles without changing the v1 bundle requirement for legacy readers.
- Keep the Measure & Inspect uncertainty model heuristic; calibrated camera geometry does not claim empirically calibrated measurement uncertainty.
- Do not change persistent reconstruction scene schemas in this step. Scene/session persistence remains Phase 9.1H.

## Distortion safety

Object point projection remains a pinhole model. Therefore non-zero lens distortion is removed before depth/perception. A rectification support mask is propagated to the native depth grid so remap border regions cannot become valid object geometry.

## Authority

- `CalibrationStatus.CALIBRATED` may flow into Measure & Inspect only from an explicit `CameraCalibrationProfile` carrying a calibration record ID.
- Camera calibration does not modify `DepthScale`.
- Camera calibration does not create a `ScaleReferenceProfile`.
- `certified_metrology` remains false.

## Expected gates

```bash
./scripts/run_phase9_1g_tests.sh quick
./scripts/run_phase9_1g_tests.sh integration
python scripts/run_phase9_1g_measurement_calibration_smoke.py
```

Manual validation is limited to the affected Measure & Inspect calibration card/profile sharing and one calibrated analysis run where practical.
