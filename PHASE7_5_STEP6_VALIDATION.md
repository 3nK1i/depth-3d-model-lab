# Phase 7.5 Step 6 Validation

## Scope

Phase 7.5 Step 6 implements dependency-light, backend-only measurement calculation and explainable uncertainty propagation from the committed Step 5 baseline.

The pipeline consumes matching:

- `AlignedFrameBundle`
- `ObjectPointExtractionBatchResult`
- `ObjectGeometryFitBatchResult`

It produces per-object, per-kind `MeasurementEstimate` contracts for:

- Width
- Height
- Depth
- Camera distance
- Centre X/Y/Z
- Roll, pitch and yaw
- Generic-box volume

No application UI, tracking, export writer, calibration, reconstruction or production shape-specific fitter is activated.

## Architecture summary

The Step 6 pipeline performs:

1. Frame and observation association validation.
2. Stable oriented-box selection where orientation is identifiable.
3. Camera-aligned or axis-aligned fallback where orientation is ambiguous.
4. Semantic width/height/depth mapping against camera axes.
5. Metric-scale eligibility checks.
6. Optional explicit scale-recovery application for relative-depth geometry.
7. Deterministic repeated fitting using percentile, point-subset, depth and intrinsics perturbations.
8. Component-wise uncertainty estimation.
9. Per-measurement reliability assessment.
10. Per-measurement availability, unavailability or rejection.

Uncertainty uses `UncertaintyBasis.HEURISTIC_PERTURBATION` and therefore does not claim a coverage probability or a statistically calibrated 95% interval.

The required display qualification remains:

```text
Estimated Uncertainty — Not Calibrated
```

## Per-measurement eligibility

Eligibility is authoritative for each measurement kind.

Examples implemented:

- Left/right border contact suppresses width, but not camera distance.
- Top/bottom border contact suppresses height, but not width.
- Any border contact suppresses generic volume.
- Ambiguous orientation suppresses roll/pitch/yaw without suppressing dimensions.
- Relative depth without successful scale recovery suppresses all metric quantities.
- Stable orientation remains available from relative geometry because it is scale-invariant.
- Synthetic and unknown depth scales do not produce available measurements.
- Excessive uncertainty rejects only the affected measurement kind.

## Uncertainty components

Applicable components are exported independently:

- Depth uncertainty
- Mask-boundary uncertainty
- Intrinsics uncertainty
- Point-cloud outlier sensitivity
- Fitting sensitivity
- Scale-recovery uncertainty
- Border-contact uncertainty
- Occlusion warning uncertainty

The authoritative stored values remain:

- `value_si`
- `expanded_uncertainty_si`

Bounds are derived and contract-validated.

## Files added

```text
core/measurement/computation_contracts.py
core/measurement/eligibility.py
core/measurement/factories.py
core/measurement/geometry.py
core/measurement/perturbation.py
core/measurement/pipeline.py
core/measurement/policy.py
core/measurement/reliability.py
core/measurement/uncertainty.py
scripts/run_measurement_uncertainty_smoke.py
scripts/run_phase7_5_step6_tests.sh
tests/integration/test_fitting_measurement_computation.py
tests/measurement_computation_helpers.py
tests/stress/test_measurement_multi_object_stress.py
tests/stress/test_measurement_perturbation_stress.py
tests/stress/test_measurement_repeated_frames_stress.py
tests/unit/test_measurement_geometry.py
tests/unit/test_measurement_uncertainty_pipeline.py
tests/unit/test_phase75_step6_backward_compatibility.py
PHASE7_5_STEP6_VALIDATION.md
```

## Files modified

```text
core/measurement/__init__.py
core/measurement/contracts.py
```

`core/measurement/__init__.py` adds lazy Step 6 exports to avoid circular imports with the existing object-geometry contracts.

`core/measurement/contracts.py` receives append-only rejection enum values and corrects scale validation so scale-invariant orientation may remain available for relative-depth geometry. Existing field ordering and existing enum values remain unchanged.

## Files removed

None.

## Installation

From the repository root:

```bash
unzip -o ~/Downloads/eia_depth_phase7_5_step6_complete_files.zip -d .
chmod +x scripts/run_measurement_uncertainty_smoke.py scripts/run_phase7_5_step6_tests.sh
```

No new dependency installation is required.

## Compile validation executed

```bash
python -m compileall -q app core scripts tests
```

Result: passed.

## Focused Step 6 gate executed

```bash
./scripts/run_phase7_5_step6_tests.sh quick
```

Result:

```text
19 passed
```

## Focused integration gate executed

```bash
./scripts/run_phase7_5_step6_tests.sh integration
```

Result:

```text
1 passed
```

## Step 6 stress sanity executed

```bash
python -m pytest -q \
  tests/stress/test_measurement_repeated_frames_stress.py \
  tests/stress/test_measurement_multi_object_stress.py \
  tests/stress/test_measurement_perturbation_stress.py
```

Result:

```text
3 passed
```

## Earlier Phase 7.5 focused regressions executed

```text
Step 1 quick: 37 passed
Step 2 quick: 34 passed
Step 3 quick: 39 passed
Step 4 quick: 32 passed
Step 5 quick: 21 passed
```

## Critical legacy depth/IPC regression executed

```bash
python -m pytest -q \
  tests/unit/test_model_output_contract.py \
  tests/unit/test_model_registry.py \
  tests/unit/test_runtime_messages.py \
  tests/unit/test_geometry_projection.py \
  tests/unit/test_snapshot_export.py
```

Result:

```text
50 passed
```

## Existing measurement contract regression executed

```text
9 passed
```

## Full split-process regression

The command remains:

```bash
./scripts/run_phase7_5_step6_tests.sh full
```

It executes these groups in separate pytest processes:

- `tests/unit`
- `tests/ui`
- `tests/integration`
- `tests/stress`

The full gate was attempted in the delivery container. Collection stopped because the container does not have PyQt6 or VisPy. This is an environment limitation, not a Step 6 test failure.

The full gate must be run in the configured Python 3.11/3.12 `depth` environment before commit.

## Standalone smoke validation executed

```bash
python scripts/run_measurement_uncertainty_smoke.py \
  --output /tmp/phase7_5_step6_measurements.png
```

Result:

```text
status=success available=11 total=11
```

The synthetic validation produced all 11 requested measurements with deterministic heuristic uncertainty and no coverage-probability claim.

## Focused manual validation

Run only:

1. Launch the application with `python main.py`.
2. Confirm Live 3D, Stable Video, Snapshot and Benchmark are unchanged.
3. Confirm Reconstruction remains disabled.
4. Confirm no measurement controls or values appear in the UI yet.
5. Run one Development Mock Live 3D inference.
6. Run one Snapshot export.
7. Close the application and confirm no worker remains.
8. Run the Step 6 smoke script and inspect its measurement/uncertainty panel.

Process check:

```bash
ps aux | grep -E \
  'main.py|inference_worker|perception_worker|depth_3d_model_lab' \
  | grep -v grep
```

## Acceptance conditions

Before commit, confirm:

- Compile validation passes.
- Step 6 quick and integration gates pass.
- Full split-process regression passes in the target environment.
- Focused manual validation passes.
- No checkpoint, output, log or smoke image is staged.
- Version remains `0.7.0`.
- Reconstruction remains disabled.

## Git commands

```bash
git status --short
git diff --check
git add -A
git diff --cached --check
git diff --cached --name-only
git commit -m "feat(measurement): add Phase 7.5 measurements and uncertainty"
git push
```

No version bump, tag or release is required for Step 6.
