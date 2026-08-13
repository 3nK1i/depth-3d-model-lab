# Phase 9.1D Validation — Automatic Checkerboard & ChArUco Calibration

Phase 9.1D adds a non-UI automatic camera-calibration engine on top of the
Phase 9.1A–C calibration contracts. It does not change OpenCV packages or any
external reconstruction runtime.

## Scope and authority

- Checkerboard calibration uses OpenCV `calib3d` target detection, subpixel
  refinement/fallback, `calibrateCamera`, and explicit per-view reprojection
  evaluation.
- ChArUco support is feature-detected from the active `cv2` build. The engine
  supports the modern `CharucoDetector` path and a legacy detection fallback;
  it does not depend on deprecated build-specific `calibrateCameraCharuco`
  wrappers.
- All calibration images in one solve must have the same resolution.
- Target dimensions are explicit inputs. No board geometry is hardcoded into
  the production engine.
- Reprojection RMS, mean, maximum, accepted-view count, rejected-view count,
  and per-view diagnostics are preserved.
- Acceptance thresholds are an explicit `AutomaticCalibrationQualityPolicy`.
  Without a policy, an automatic profile remains `unvalidated`; failed policy
  checks produce `rejected`, never a silently accepted profile.
- Checkerboard/ChArUco target dimensions may scale target/extrinsic coordinates
  but do not establish scene metric scale. Metric authority remains a separate
  `ScaleReferenceProfile` concern and `certified_metrology` remains false.
- Phase 8 live calibration, 9.1A profile serialization, 9.1B profile
  application, 9.1C reusable storage, UI, measurement, and external runtimes
  remain unchanged.

## Environment gate

The production `depth` environment is authoritative. Run the smoke before
commit. If it reports `BLOCKED_CHARUCO_CAPABILITY`, do not install or upgrade
OpenCV automatically; capture the reported distribution/version/capabilities
and resolve the package decision explicitly.

## Focused validation

```bash
./scripts/run_phase9_1d_tests.sh quick
./scripts/run_phase9_1d_tests.sh integration
python scripts/run_phase9_1d_automatic_calibration_smoke.py
git diff --check
```
