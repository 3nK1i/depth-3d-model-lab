# Depth & 3D Model Lab
## Phase 7.5 Step 3 — RGB, Mask, Depth and Intrinsics Alignment

## Status

Implementation completed against the committed Phase 7.5 Step 2 source archive.

The application version remains:

```text
0.7.0 — Phase 7 Stable
```

This step adds a contract-only/runtime-independent alignment layer. It does not activate perception or measurement controls in the application UI.

## Architecture summary

The new `core/alignment` package joins matching RGB, perception and depth results using the existing immutable frame identity:

```text
session ID + frame ID + source timestamp + source image size
```

The native depth grid is authoritative:

```text
RGB and selected object masks -> explicit pixel transforms -> native depth grid
Native depth, confidence, validity and intrinsics -> preserved without resampling
```

Key properties:

- Depth is never resized by the default Step 3 pipeline.
- Intrinsics remain tied to the native depth pixel space.
- RGB uses area interpolation for pure reductions and linear interpolation otherwise.
- Boolean masks always use nearest-neighbour resampling.
- Approved manual masks remain authoritative.
- Overlapping object masks remain independent.
- Detection boxes are transformed through their four corners and clipped using `[x_min, y_min, x_max, y_max)` semantics.
- Crop, resize and padding provenance must agree with the authoritative 3×3 transform.
- Missing, unsupported or empty masks produce explicit partial results and warnings; the pipeline does not fabricate measurements or geometry.
- Model-predicted and estimated intrinsics remain `Not Calibrated`.
- No confidence threshold or object-point filtering is applied in Step 3.

## Files added

```text
core/alignment/__init__.py
core/alignment/contracts.py
core/alignment/factories.py
core/alignment/intrinsics.py
core/alignment/pipeline.py
core/alignment/resampling.py
core/alignment/transforms.py
core/alignment/validation.py

scripts/run_alignment_smoke.py
scripts/run_phase7_5_step3_tests.sh

tests/unit/test_alignment_contracts.py
tests/unit/test_alignment_factories.py
tests/unit/test_alignment_intrinsics.py
tests/unit/test_alignment_pipeline.py
tests/unit/test_alignment_resampling.py
tests/unit/test_alignment_transforms.py
tests/unit/test_alignment_validation.py
tests/unit/test_phase75_step3_backward_compatibility.py

tests/integration/test_depth_perception_alignment.py

tests/stress/test_alignment_repeated_frames_stress.py
tests/stress/test_alignment_resolution_stress.py

PHASE7_5_STEP3_VALIDATION.md
```

## Files modified

None.

All existing application, runtime, worker, depth-model, perception, geometry, UI and reconstruction files remain unchanged.

## Files removed

None.

## Installation

After downloading the ZIP and checksum into `~/Downloads`:

```bash
cd ~/Downloads
sha256sum -c eia_depth_phase7_5_step3_complete_files.zip.sha256
```

Expected:

```text
eia_depth_phase7_5_step3_complete_files.zip: OK
```

Install from the repository root:

```bash
cd /home/eiatech/Desktop/depth_3d_model_lab

git status --short

unzip -o \
  ~/Downloads/eia_depth_phase7_5_step3_complete_files.zip \
  -d .

chmod +x \
  scripts/run_alignment_smoke.py \
  scripts/run_phase7_5_step3_tests.sh
```

Step 3 adds no new Python dependency. It uses NumPy and OpenCV already required by the application.

## Compile validation

```bash
python -m compileall -q app core scripts tests
```

No output means compilation passed.

## Automated test commands

### Focused Step 3 gate

```bash
./scripts/run_phase7_5_step3_tests.sh quick
```

Executed in the delivery environment:

```text
39 passed
```

Coverage includes:

- Immutable and pickle-safe contracts
- Exact cross-pipeline frame identity
- Identity, resize, crop and padding transforms
- Authoritative transform disagreement rejection
- Landscape, portrait and square inputs
- Nonuniform resize warnings
- RGB resampling
- Nearest-neighbour boolean masks
- No fractional mask output
- RGB support masks
- Detection-box transformation and clipping
- Manual-mask authority
- Overlapping-mask preservation
- Missing selected masks
- Empty aligned masks
- Schema-ready non-dense mask handling
- Depth validity composition
- Confidence and uncertainty semantics
- Native-intrinsics preservation
- Intrinsics transformation
- Relative-depth semantic preservation
- Read-only result arrays
- No input mutation
- No PyTorch, Ultralytics, Qt or VisPy import
- Existing depth registry order and IPC constructor compatibility
- Version remains `0.7.0`

### Focused integration gate

```bash
./scripts/run_phase7_5_step3_tests.sh integration
```

Executed in the delivery environment:

```text
1 passed
```

This joins the dependency-free Step 2 perception mock with an existing `InferenceResult`, then aligns both outputs without changing either worker or IPC contract.

### Previous Phase 7.5 contract gates

```bash
./scripts/run_phase7_5_step1_tests.sh quick
./scripts/run_phase7_5_step2_tests.sh quick
```

Executed in the delivery environment:

```text
Step 1: 37 passed
Step 2: 34 passed
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

### Alignment stress sanity

```bash
python -m pytest -q \
  tests/stress/test_alignment_repeated_frames_stress.py \
  tests/stress/test_alignment_resolution_stress.py
```

Executed in the delivery environment:

```text
6 passed
```

### Full split-process regression

Run in the configured project Python 3.11/3.12 `depth` environment:

```bash
./scripts/run_phase7_5_step3_tests.sh full
```

The script launches these groups in separate pytest processes:

```text
tests/unit
tests/ui
tests/integration
tests/stress
```

The full gate was attempted in the delivery container but cannot be claimed there because that container uses unsupported Python 3.13.5 and does not contain PyQt6 or VisPy. The focused Step 3, prior Step 1/2, critical compatibility and alignment stress gates listed above all passed.

## Standalone alignment smoke validation

Run with a real image:

```bash
python scripts/run_alignment_smoke.py \
  --image ~/Downloads/depth-test.png \
  --output /tmp/phase7_5_step3_alignment.png
```

Expected console form:

```text
status=success observations=1 target=<width>x<height> output=/tmp/phase7_5_step3_alignment.png
```

The debug image contains:

- Original RGB with the source selected mask
- RGB and mask aligned to the native depth grid
- Native synthetic depth preview
- Aligned boolean mask

The delivery-environment smoke run completed successfully and created the output image. This is a dependency-free alignment smoke test, not a claim of real model or camera calibration accuracy.

## Focused manual validation

Manual validation remains minimal because Step 3 adds no application UI or runtime orchestration:

1. Launch the application:

   ```bash
   python main.py
   ```

2. Confirm Live 3D, Stable Video, Snapshot and Benchmark are unchanged.
3. Confirm Reconstruction remains disabled.
4. Confirm no perception, alignment or measurement controls appear.
5. Run one Development Mock Live 3D inference.
6. Run one Snapshot export.
7. Close the application and confirm no worker remains:

   ```bash
   ps aux | grep -E \
     'main.py|inference_worker|perception_worker|depth_3d_model_lab' \
     | grep -v grep
   ```

8. Run the standalone alignment smoke command.
9. Inspect its generated debug image and confirm the selected mask remains spatially aligned after conversion to the depth grid.

Real Qt manual validation was not executed in the delivery container and is not claimed.

## Pre-commit review

```bash
git status --short
git diff --check
git diff --stat
git ls-files --others --exclude-standard | sort
```

Confirm:

- No existing runtime or UI file is modified.
- No generated image, model checkpoint, output or log is staged.
- `core/version.py` and `pyproject.toml` remain at version `0.7.0`.
- Reconstruction remains disabled.

## Git commit and push

Only after compilation, focused gates, full split-process regression and focused manual validation pass:

```bash
git status --short
git add -A
git status --short
git commit -m "feat(alignment): add Phase 7.5 RGB mask depth alignment"
git push
```

No version bump, annotated tag or release is created for Step 3.
