# Phase 7.5 Real Validation — No-Confidence Extraction and 3D Box Visual Fix

## Scope

This corrective delivery fixes the real-image validation path without changing the Qt UI, depth worker, perception worker, measurement algorithms, model adapters, version, or reconstruction state.

## Root cause

Depth Pro reports `ConfidenceKind.NONE` and no confidence array. Step 3 previously represented that state with an all-false `confidence_valid_mask`. Step 4 correctly intersected candidate object pixels with that map, which removed every pixel and rejected every object as `too_few_valid_points`.

The corrected semantic is:

- confidence/uncertainty map present: validate its finite semantic support;
- `ConfidenceKind.NONE`: no confidence-based restriction, therefore all pixels remain confidence-eligible and depth validity remains authoritative.

## Files replaced

- `core/alignment/validation.py`
- `scripts/run_real_measurement_validation.py`
- `tests/unit/test_alignment_validation.py`
- `tests/unit/test_real_measurement_validation_script.py`

## File added

- `tests/integration/test_no_confidence_alignment_extraction.py`

## Validation report improvements

The standalone real-validation runner now produces:

- `rgb_detections_masks.png`
- `depth_preview.png`
- `aligned_geometry_boxes.png`
- `object_NNN_geometry_measurements.png`
- `overview.png`
- `measurements.json`

The visuals include:

- strict axis-aligned box;
- robust camera-aligned box;
- robust oriented box;
- highlighted preferred box;
- projected 3D boxes on the aligned depth-grid RGB image;
- camera X/Z object points with fitted box wireframes;
- extraction stage counts and rejection reasons;
- fitting status and box extents;
- measurements, uncertainty and reliability.

The JSON now includes extraction stage counts, extraction/fitting rejection reasons and complete fitted-box data.

## Validation executed in delivery environment

```text
Compile validation: passed
Targeted unit/integration correction gate: 11 passed
Step 3 quick regression: 39 passed
Step 4 quick regression: 32 passed
Step 6 quick regression: 19 passed
Real-validation script tests: 7 passed
Critical alignment/extraction/measurement integration: 4 passed
```

Real Depth Pro, Ultralytics and CUDA execution were not available in the delivery environment and are not claimed.

## Target-machine validation

```bash
python -m compileall -q core scripts tests

python -m pytest -q \
  tests/unit/test_alignment_validation.py \
  tests/integration/test_no_confidence_alignment_extraction.py \
  tests/unit/test_real_measurement_validation_script.py

./scripts/run_phase7_5_step3_tests.sh quick
./scripts/run_phase7_5_step4_tests.sh quick
./scripts/run_phase7_5_step6_tests.sh quick
./scripts/run_phase7_5_real_validation_tests.sh
```

Run the real validation again:

```bash
rm -rf /tmp/phase7_5_real_measurement

python scripts/run_real_measurement_validation.py \
  --image ~/Downloads/depth-test.png \
  --depth-model depth-pro \
  --perception-model ultralytics-yolo-seg \
  --depth-profile efficient \
  --perception-profile efficient \
  --max-objects 5 \
  --output-dir /tmp/phase7_5_real_measurement
```

Open:

```bash
google-chrome /tmp/phase7_5_real_measurement/overview.png
google-chrome /tmp/phase7_5_real_measurement/aligned_geometry_boxes.png
google-chrome /tmp/phase7_5_real_measurement/object_000_geometry_measurements.png
```

Results remain validation-only and must continue to display `Estimated Uncertainty — Not Calibrated`.
