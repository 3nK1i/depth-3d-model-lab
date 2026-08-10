# Phase 7.5 Step 9 Validation

Baseline commit: `dd78e20d0f8a6684cfd9a6c60b5a0392320ce177`

## Scope

Step 9 adds deterministic, non-destructive inspection exports without rerunning depth or perception:

- Selected-object and all-object export controls
- Isolated export subprocess to keep the UI responsive
- RGB, annotated RGB and depth-preview PNG files
- Automatic and approved/current masks preserved separately
- Raw and filtered colored PLY point clouds
- All three 3D box payloads and measurements in JSON
- Flat measurement CSV including rejected values and reasons
- Model, camera and session provenance
- HTML and PDF inspection reports
- SHA-256 manifest for every exported file
- Explicit `Estimated Uncertainty - Not Calibrated` and non-certified-metrology notices

## Validation completed in the delivery environment

- Input archive checksum: passed
- Python compilation: passed
- Step 9 quick gate: 3 passed
- Step 9 integration gate: 1 passed
- Step 1-8 focused quick regressions: 227 passed
- Development mock export smoke: passed, 2 objects / 23 files
- PDF render verification: passed, 3 pages rendered at 150 DPI
- Fresh-baseline overlay integrity: passed

PyQt6 and VisPy are unavailable in the delivery container, so the focused UI gate and complete split-process regression must be run in the configured `depth` environment.

## Safety and compatibility

- Export reads the already completed inspection bundle; no model is loaded or rerun.
- The original automatic mask and current approved mask are stored separately.
- Existing inspection bundles remain schema version 1 and load unchanged.
- Rejected measurements remain rejected in CSV, JSON, HTML and PDF.
- Export output is atomic: a partial destination is removed on failure.
- Reconstruction remains disabled.
- Application version remains `0.7.0 - Phase 7 Stable` until Step 10 release.
