# EIA TECH Depth & 3D Model Lab
## Phase 7.5 Step 4 — Object-Specific 3D Point Extraction

## Status

Implementation completed against the committed Phase 7.5 Step 3 source archive.

The application version remains:

```text
0.7.0 — Phase 7 Stable
```

Step 4 adds a standalone, segmentation-first object-geometry layer. It does not activate perception, point-cloud inspection or measurement controls in the application UI.

## Architecture summary

The new `core/object_geometry` package consumes the authoritative Step 3 `AlignedFrameBundle` and produces an independent result for each aligned object:

```text
aligned object mask
∩ RGB support
∩ valid depth
∩ confidence/uncertainty validity
∩ optional depth range
→ selected native-depth pixels
→ direct pinhole back-projection
→ robust filtering
→ validated ObjectPointCloud
```

Key properties:

- Selected object pixels are projected directly from native depth and native intrinsics.
- The global rendered/decimated VisPy point cloud is never reused.
- Every 3D point preserves its explicit source pixel coordinate and pixel-space ID.
- Raw and filtered clouds remain separate and include explainable stage counts.
- Approved manual-mask provenance remains authoritative and is preserved in processing provenance.
- Overlapping object masks remain independent.
- Boundary erosion is conservative and is skipped when it would destroy valid support.
- Confidence and uncertainty use generic `ConfidenceKind` semantics; no model-specific threshold exists in the pipeline.
- Depth-discontinuity rejection uses robust four-neighbour median/MAD statistics.
- Statistical 3D outlier filtering uses dependency-light robust median/MAD filtering.
- Background-plane removal is available but disabled by default and requires a verified exterior-ring plane behind the object.
- Voxel clustering is deterministic and prefers the cluster associated with the projected mask centroid.
- Point-budget sampling is deterministic rather than random.
- Border contact is preserved as a warning/partial result and does not globally reject every future measurement kind.
- Relative depth produces explicit relative-model-unit geometry with `metric_ready=false`.
- Synthetic and unknown scale semantics remain explicit.
- No dimension, distance, box fit, tracking, export writer or UI feature is implemented in this step.

## Files added

```text
core/object_geometry/__init__.py
core/object_geometry/clustering.py
core/object_geometry/contracts.py
core/object_geometry/factories.py
core/object_geometry/filters.py
core/object_geometry/masks.py
core/object_geometry/pipeline.py
core/object_geometry/policy.py
core/object_geometry/projection.py
core/object_geometry/sampling.py
core/object_geometry/validation.py

scripts/run_object_point_extraction_smoke.py
scripts/run_phase7_5_step4_tests.sh

tests/object_geometry_helpers.py

tests/unit/test_object_geometry_clustering.py
tests/unit/test_object_geometry_contracts.py
tests/unit/test_object_geometry_factories.py
tests/unit/test_object_geometry_filters.py
tests/unit/test_object_geometry_masks.py
tests/unit/test_object_geometry_pipeline.py
tests/unit/test_object_geometry_projection.py
tests/unit/test_object_geometry_sampling.py
tests/unit/test_object_geometry_validation.py
tests/unit/test_phase75_step4_backward_compatibility.py

tests/integration/test_alignment_object_point_extraction.py

tests/stress/test_object_point_extraction_multi_object_stress.py
tests/stress/test_object_point_extraction_repeated_frames.py
tests/stress/test_object_point_extraction_resolution_stress.py

PHASE7_5_STEP4_VALIDATION.md
```

## Files modified

None.

All pre-existing application, runtime, worker, model, perception, alignment, geometry, measurement, tracking, UI and reconstruction files remain byte-for-byte unchanged relative to the uploaded committed Step 3 baseline.

## Files removed

None.

## Installation

After downloading the ZIP and checksum into `~/Downloads`:

```bash
cd ~/Downloads
sha256sum -c eia_depth_phase7_5_step4_complete_files.zip.sha256
```

Expected:

```text
eia_depth_phase7_5_step4_complete_files.zip: OK
```

Install from the repository root:

```bash
cd /home/eiatech/Desktop/depth_3d_model_lab

git status --short

unzip -o \
  ~/Downloads/eia_depth_phase7_5_step4_complete_files.zip \
  -d .

chmod +x \
  scripts/run_object_point_extraction_smoke.py \
  scripts/run_phase7_5_step4_tests.sh
```

Step 4 adds no new Python dependency. It uses NumPy and OpenCV already required by the application.

## Compile validation

```bash
python -m compileall -q app core scripts tests
```

No output means compilation passed.

## Automated test commands

### Focused Step 4 gate

```bash
./scripts/run_phase7_5_step4_tests.sh quick
```

Executed in the delivery environment:

```text
32 passed
```

Coverage includes:

- Immutable and pickle-safe result contracts
- Read-only arrays restored after multiprocessing/pickle round trips
- Monotonic per-stage point counts
- Missing and empty-mask rejection
- Approved manual-mask provenance
- Largest connected component selection
- Conservative boundary erosion
- RGB-support and depth-validity intersection
- Confidence and uncertainty semantics
- Explicit depth-range filtering
- Robust depth-discontinuity rejection
- Direct selected-pixel projection
- Camera axis convention `X right, Y down, Z forward`
- Explicit source pixel indices
- RGB and point-weight association
- Robust 3D outlier filtering
- Conservative background-plane acceptance and rejection
- Voxel-cluster selection and ambiguity reporting
- Deterministic point-budget sampling
- Multiple and overlapping objects
- Object-selection subsets
- Border-contact preservation
- Relative and synthetic nonmetric semantics
- No input mutation
- No PyTorch, CUDA, PyQt6, Ultralytics or VisPy import
- Existing depth and perception registry ordering
- Version remains `0.7.0`

### Focused integration gate

```bash
./scripts/run_phase7_5_step4_tests.sh integration
```

Executed in the delivery environment:

```text
1 passed
```

This validates the complete Step 3 aligned-bundle to Step 4 object-point-cloud path.

### Previous Phase 7.5 focused gates

```bash
./scripts/run_phase7_5_step1_tests.sh quick
./scripts/run_phase7_5_step2_tests.sh quick
./scripts/run_phase7_5_step3_tests.sh quick
```

Executed in the delivery environment:

```text
Step 1: 37 passed
Step 2: 34 passed
Step 3: 39 passed
```

### Critical depth and IPC compatibility gate

```bash
python -m pytest -q \
  tests/unit/test_model_output_contract.py \
  tests/unit/test_model_registry.py \
  tests/unit/test_runtime_messages.py \
  tests/unit/test_geometry_projection.py \
  tests/unit/test_snapshot_export.py \
  tests/unit/test_phase75_step2_backward_compatibility.py
```

Executed in the delivery environment:

```text
55 passed
```

### Step 4 stress sanity

```bash
python -m pytest -q \
  tests/stress/test_object_point_extraction_repeated_frames.py \
  tests/stress/test_object_point_extraction_resolution_stress.py \
  tests/stress/test_object_point_extraction_multi_object_stress.py
```

Executed in the delivery environment:

```text
3 passed
```

The stress tests cover deterministic repeated extraction, landscape/portrait/square resolutions and 20 overlapping objects.

### Full split-process regression

Run in the configured EIA TECH Python 3.11/3.12 `depth` environment:

```bash
./scripts/run_phase7_5_step4_tests.sh full
```

The runner launches each group in a separate pytest process:

```text
tests/unit
tests/ui
tests/integration
tests/stress
```

The full gate was attempted in the delivery container but cannot be claimed there because that container uses unsupported Python 3.13.5 and does not contain PyQt6 or VisPy. The focused Step 4, prior Step 1/2/3, critical compatibility and Step 4 stress gates listed above passed.

## Standalone extraction smoke validation

Run:

```bash
python scripts/run_object_point_extraction_smoke.py \
  --output /tmp/phase7_5_step4_extraction.png
```

Executed in the delivery environment:

```text
status=success raw_points=21770 filtered_points=21430 metric_ready=true
```

The deterministic smoke scene includes:

- A known object mask
- Metric object depth
- A background-depth contamination patch
- Isolated depth spikes
- Native estimated intrinsics

The generated debug image shows:

- Synthetic RGB and selected mask
- Metric depth with contamination and spikes
- Raw projected object pixels
- Filtered object points in a top-down X/Z view

This image is a validation artifact only; it is not a production export writer.

Open it with:

```bash
firefox /tmp/phase7_5_step4_extraction.png
```

or:

```bash
google-chrome /tmp/phase7_5_step4_extraction.png
```

## Focused manual validation

Step 4 adds no application UI feature. Perform only:

1. Launch the application:

   ```bash
   python main.py
   ```

2. Confirm Live 3D, Stable Video, Snapshot and Benchmark remain unchanged.
3. Confirm Reconstruction remains disabled.
4. Confirm no object-point, detection or measurement controls appear.
5. Run one Development Mock Live 3D inference.
6. Run one Snapshot export.
7. Close the application and confirm no worker remains:

   ```bash
   ps aux | grep -E \
     'main.py|inference_worker|perception_worker|depth_3d_model_lab' \
     | grep -v grep
   ```

8. Run and inspect the standalone Step 4 smoke visualization.

## Git commands after all validation passes

```bash
git status --short
git diff --check
git add -A
git status --short
git diff --cached --check
git diff --cached --name-only

git commit -m "feat(object-geometry): add Phase 7.5 object point extraction"
git push
```

Do not create a version bump, tag or release for Step 4.

## Step 4 boundaries preserved

This delivery does not implement or activate:

- Production application orchestration for perception/alignment/extraction
- Object-point visualization in the main UI
- Raw-versus-filtered inspection controls
- 3D boxes or principal axes
- Shape fitting
- Dimensions or camera distance
- Measurement uncertainty calculation
- Tracking
- Measurement exports or reports
- Camera calibration
- SfM, SLAM or reconstruction
- Scene or multi-camera fusion
- Digital Twin functionality

Those remain assigned to later Phase 7.5 and Phase 8 steps.
