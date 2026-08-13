# Phase 9.1F — Professional Camera Calibration UI

## Scope

Phase 9.1F exposes the Phase 9.1A–E camera-calibration foundation inside the existing Reconstruction product UI without adding a new top-level mode.

The UI provides:

- a dedicated `Camera Calibration` workspace/dialog;
- Automatic / Manual / Import Profile paths;
- checkerboard and ChArUco image-set selection;
- explicit automatic-calibration quality thresholds and reprojection result presentation;
- manual width/height, `fx`, `fy`, `cx`, `cy`, Brown-Conrady `k1/k2/p1/p2/k3` entry;
- native Phase 9 and legacy Phase 8 JSON import;
- reusable filesystem-backed saved profiles;
- profile selection/clear controls in the existing `AUTHORITY & SAFETY` disclosure;
- safe Phase 9.1B same-aspect full-frame application when a selected profile is used for live reconstruction;
- an explicit Test Mode fallback for provisional non-metric live testing.

## Authority / compatibility rules

- Camera calibration never qualifies scene metric scale.
- Manual entry remains `UNVALIDATED` until separately qualified.
- Legacy Phase 8 JSON remains `LEGACY_COMPATIBLE`.
- Automatic calibration uses the Phase 9.1D validation state and quality evidence unchanged.
- The selected profile is applied through `apply_camera_calibration_profile`; unsafe resolution changes remain rejected.
- Existing Phase 8 legacy calibration JSON loading is retained as a fallback compatibility path.
- `certified_metrology` is not promoted by this UI.
- No calibration, scale, measurement, persistence, release-metadata, or external-runtime core contract is changed.

## Deferred intentionally

Phase 9.1F does not yet persist the active calibration/metric-evidence selection into persistent reconstruction scenes and does not author scene-specific metric evidence in the UI. That integration belongs to the next Phase 9.1 step after this UI is qualified.

## Validation

Run:

```bash
./scripts/run_phase9_1f_tests.sh quick
./scripts/run_phase9_1f_tests.sh integration
python scripts/run_phase9_1f_calibration_ui_preview.py
```

Manual validation is intentionally limited to the affected UI:

1. At 1366×768, open Reconstruction and confirm the right sidebar has no horizontal scrolling/clipping.
2. Open `Manage Calibration` from `AUTHORITY & SAFETY`.
3. Verify Automatic / Manual / Import Profile are clear and usable with vertical scrolling only.
4. Confirm Camera Calibration and Metric Scale are displayed as separate states.
5. Build a manual profile and confirm it is shown as unvalidated and metric scale remains not qualified.
6. Confirm Test Mode is visibly provisional/non-metric when invoked from Live Start.

Do not run a full manual application regression for this substep.
