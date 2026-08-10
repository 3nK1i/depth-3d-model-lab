# Phase 7.5 Step 5 Validation

## Scope

Step 5 adds dependency-light robust object-frame, principal-axis, 3D-box and generic-box shape fitting on top of the committed Step 4 object-specific point-cloud contracts.

No UI, worker, depth-model, perception-model, alignment, reconstruction, version, or release behavior is activated or changed.

## Files added

- `core/object_geometry/fitting_contracts.py`
- `core/object_geometry/fitting_policy.py`
- `core/object_geometry/robust_statistics.py`
- `core/object_geometry/principal_axes.py`
- `core/object_geometry/frames.py`
- `core/object_geometry/boxes.py`
- `core/object_geometry/shape_fitting.py`
- `core/object_geometry/fitting_pipeline.py`
- `core/object_geometry/fitting_validation.py`
- `core/object_geometry/fitting_factories.py`
- `scripts/run_object_geometry_fitting_smoke.py`
- `scripts/run_phase7_5_step5_tests.sh`
- `tests/object_geometry_fitting_helpers.py`
- `tests/unit/test_object_geometry_fitting.py`
- `tests/unit/test_phase75_step5_backward_compatibility.py`
- `tests/integration/test_extraction_object_geometry_fitting.py`
- `tests/stress/test_object_geometry_fitting_stress.py`

## Files modified

- `core/object_geometry/__init__.py` — append-only public exports for Step 5.

## Files removed

None.

## Implemented behavior

- Robust weighted/unweighted centre estimation
- Robust tail trimming for fitting stability
- Deterministic right-handed PCA axes
- Eigenvalue and explained-variance diagnostics
- Linear, planar, point-like and volumetric degeneracy classification
- Per-axis orientation stability
- Object-local frame and object-to-camera rigid transform
- Strict axis-aligned box
- Robust camera-aligned percentile box
- Robust PCA-oriented percentile box
- Preferred-box fallback policy
- Generic box shape-model output
- Explicit metric-readiness propagation
- Explicit rejection for insufficient points or failed fitting
- Pickle-safe, read-only numerical arrays

## Validation executed in the delivery environment

Environment limitations:

- Python 3.13.5
- PyQt6 unavailable
- VisPy unavailable
- No target RTX GPU validation

Passed:

```text
python -m compileall -q app core scripts tests
Step 5 quick gate:          21 passed
Step 5 integration gate:     1 passed
Step 5 stress sanity:         3 passed
Step 5 smoke script:          passed
```

Smoke result:

```text
status=partial
preferred_box=oriented
orientation_stability=ambiguous
```

The `partial` status is intentional for the synthetic sample: the dominant axis is stable, while the two smaller PCA axes are not separated strongly enough to claim fully stable orientation. The oriented box is retained with explicit ambiguity, while the camera-aligned fallback remains available.

The complete legacy unit/UI suite was not claimed in this delivery container because collection requires PyQt6 and VisPy. Run the split-process full gate in the configured Python 3.11/3.12 `depth` environment.

## Commands

Compile:

```bash
python -m compileall -q app core scripts tests
```

Focused gate:

```bash
./scripts/run_phase7_5_step5_tests.sh quick
```

Integration:

```bash
./scripts/run_phase7_5_step5_tests.sh integration
```

Full split-process regression:

```bash
./scripts/run_phase7_5_step5_tests.sh full
```

Smoke:

```bash
python scripts/run_object_geometry_fitting_smoke.py \
  --output /tmp/phase7_5_step5_fitting.png
```

## Focused manual validation

1. Launch `python main.py`.
2. Confirm Live 3D, Stable Video, Snapshot and Benchmark are unchanged.
3. Confirm Reconstruction remains disabled.
4. Confirm no Step 5 boxes or measurement controls appear.
5. Run Development Mock Live 3D once.
6. Run one Snapshot export.
7. Close the application and confirm no worker remains.
8. Run the Step 5 smoke script and inspect the diagnostic image.

## Git

After all tests and manual checks pass:

```bash
git status --short
git add -A
git commit -m "feat(object-geometry): add Phase 7.5 robust object fitting"
git push
```

No version bump, tag, release, production UI activation, or reconstruction activation belongs to Step 5.
