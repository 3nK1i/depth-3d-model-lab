# Phase 8.1 Validation — Reconstruction Foundation Contracts

## Scope

Phase 8.1 adds a dependency-light `core/reconstruction/` foundation for later
SfM, SLAM, loop closure, dense fusion, meshing, exports, and UI integration.
It does not run a real reconstruction backend and does not enable the
Reconstruction mode.

The implementation defines:

- reconstruction input, frame, keyframe, session, trajectory, sparse-map, and
  provenance contracts;
- canonical metres, camera-to-world pose semantics and explicit coordinate
  conventions;
- pose inversion, composition, quaternion conversion, and rigid-transform
  validation;
- camera intrinsics and distortion declarations;
- structured lifecycle, progress, warning, failure, pause, resume, and
  cancellation events;
- cooperative cancellation and pause controls;
- backend capability declarations and typed interfaces for offline SfM, live
  SLAM, dense fusion, and mesh generation;
- Efficient, Balanced, and High Quality reconstruction profiles;
- bounded RTX 2060 / RTX 5090 CPU and GPU memory policies;
- checkpoint, resume-compatibility, schema-version, and provenance contracts;
- stable deterministic JSON serialization for intermediate metadata.

## Non-goals

Phase 8.1 does not add feature detection, matching, SfM execution, bundle
adjustment, SLAM tracking, loop closure, fusion, meshing, textures, rendering,
exports, a reconstruction worker, or Reconstruction UI controls.

## Files added

### Production foundation

- `core/reconstruction/__init__.py`
- `core/reconstruction/backends.py`
- `core/reconstruction/camera.py`
- `core/reconstruction/checkpoint.py`
- `core/reconstruction/contracts.py`
- `core/reconstruction/control.py`
- `core/reconstruction/errors.py`
- `core/reconstruction/events.py`
- `core/reconstruction/poses.py`
- `core/reconstruction/profiles.py`
- `core/reconstruction/provenance.py`
- `core/reconstruction/serialization.py`
- `core/reconstruction/session.py`
- `core/reconstruction/sparse.py`
- `core/reconstruction/trajectory.py`
- `core/reconstruction/validation.py`

### Automated validation

- `tests/reconstruction_helpers.py`
- `tests/unit/test_reconstruction_camera_poses.py`
- `tests/unit/test_reconstruction_contracts_session.py`
- `tests/unit/test_reconstruction_sparse_events.py`
- `tests/unit/test_reconstruction_profiles_backends.py`
- `tests/unit/test_reconstruction_checkpoint_serialization.py`
- `tests/unit/test_phase8_1_backward_compatibility.py`
- `tests/integration/test_reconstruction_contract_lifecycle.py`
- `tests/stress/test_reconstruction_contracts_stress.py`
- `scripts/run_phase8_1_tests.sh`
- `scripts/run_reconstruction_contracts_smoke.py`

## Validation performed while preparing the package

- Python compilation of all Phase 8.1 production, test, and smoke files: PASS
- Phase 8.1 quick suite plus focused backward-compatibility checks: 67 passed
- Phase 8.1 integration and meaningful trajectory stress checks: 3 passed
- Non-UI reconstruction contracts smoke: PASS
- Synthetic RTX 2060 6 GB policy: Efficient
- Synthetic RTX 5090 32 GB policy: High Quality
- Metadata JSON serialize/restore round trip: PASS
- Package content, whitespace, executable flags, and SHA-256 verification: PASS

The complete historical application suite was not executed in the package
preparation container because PyQt6 and VisPy are unavailable there. Run the
full suite in the project `depth` conda environment.

## Installation

```bash
cd ~/Downloads
sha256sum -c eia_depth_phase8_1_reconstruction_foundation.zip.sha256

cd /home/eiatech/Desktop/depth_3d_model_lab

git branch --show-current
git status --short

unzip -o \
  ~/Downloads/eia_depth_phase8_1_reconstruction_foundation.zip \
  -d .

chmod +x scripts/run_phase8_1_tests.sh
chmod +x scripts/run_reconstruction_contracts_smoke.py
```

Expected branch:

```text
phase8-reconstruction
```

## Compilation

```bash
python -m compileall -q \
  core/reconstruction \
  tests/reconstruction_helpers.py \
  tests/unit/test_reconstruction_*.py \
  tests/unit/test_phase8_1_backward_compatibility.py \
  tests/integration/test_reconstruction_contract_lifecycle.py \
  tests/stress/test_reconstruction_contracts_stress.py \
  scripts/run_reconstruction_contracts_smoke.py
```

## Focused tests

```bash
./scripts/run_phase8_1_tests.sh quick
```

Expected package-preparation result:

```text
67 passed
```

## Integration and stress tests

```bash
./scripts/run_phase8_1_tests.sh integration
```

Expected package-preparation result:

```text
3 passed
```

## Non-UI smoke

```bash
python scripts/run_reconstruction_contracts_smoke.py
```

Optional metadata output path:

```bash
python scripts/run_reconstruction_contracts_smoke.py \
  --output /tmp/phase8_1_reconstruction_metadata.json
```

Expected summary begins with:

```text
Phase 8.1 reconstruction contracts smoke: PASS
```

The smoke is synthetic. It performs no downloads, opens no UI, and runs no
real SfM, SLAM, fusion, or mesh backend.

## Full regression

```bash
./scripts/run_phase8_1_tests.sh full
```

The full mode runs the existing groups:

```text
tests/unit
tests/ui
tests/integration
tests/stress
```

The two previously documented optional real Ultralytics/SAM integration skips
may remain expected on systems where those hardware smoke tests are deferred.

## Targeted manual validation

Only the following manual checks are required for Phase 8.1:

1. Start the application normally.
2. Confirm the header still displays `v0.7.5`.
3. Confirm Live 3D and Measure & Inspect still open normally.
4. Confirm Reconstruction remains visible but disabled.
5. Open one image and perform one small existing depth-generation sanity check.
6. Close the application normally.

No manual SfM, SLAM, fusion, or mesh validation belongs to Phase 8.1.

## Git review and commit

```bash
git diff --check
git status --short
git diff --stat

git add -A

git diff --cached --check
git diff --cached --stat
git diff --cached --name-only
```

Confirm that only the Phase 8.1 files listed above are staged and that no
models, checkpoints, generated output, cache, report, log, or profiling files
are included.

Commit and push:

```bash
git commit -m "feat(reconstruction): add Phase 8 foundation contracts"
git push
```

Do not create a tag and do not merge into `main` during this step.
