# Phase 8.2 Validation — Reconstruction Ingestion and Keyframes

## Scope

Phase 8.2 adds an isolated, non-UI offline ingestion layer for image sequences and recorded videos. It validates and orders decoded frames, preserves source indices and timestamps, applies deterministic sampling, selects initial keyframes, and bounds decoded-image memory. USB and RTSP contracts are declared for future live integration, but live reconstruction is not activated.

Phase 8.2 does not implement feature extraction, feature matching, camera pose estimation, SfM, SLAM, dense fusion, meshing, or Reconstruction UI activation.

## Files

Modified:

- `core/reconstruction/__init__.py`

Added:

- `core/reconstruction/cache.py`
- `core/reconstruction/ingestion.py`
- `core/reconstruction/keyframes.py`
- `core/reconstruction/quality.py`
- `core/reconstruction/sampling.py`
- `core/reconstruction/sources.py`
- `scripts/run_phase8_2_tests.sh`
- `scripts/run_reconstruction_ingestion_smoke.py`
- `tests/reconstruction_ingestion_helpers.py`
- `tests/unit/test_reconstruction_ingestion_sources.py`
- `tests/unit/test_reconstruction_ingestion_quality_sampling.py`
- `tests/unit/test_reconstruction_ingestion_cache.py`
- `tests/unit/test_reconstruction_ingestion_pipeline.py`
- `tests/unit/test_phase8_2_backward_compatibility.py`
- `tests/integration/test_reconstruction_ingestion_lifecycle.py`
- `tests/stress/test_reconstruction_ingestion_stress.py`
- `PHASE8_2_VALIDATION.md`

## Implemented behavior

- Natural ordering for image directories, including `frame2` before `frame10`.
- Optional preservation of explicit caller ordering.
- Lazy OpenCV decoding for image sequences and recorded videos.
- Video presentation timestamps with deterministic synthetic fallback.
- Corrupt-file reporting without aborting the complete sequence.
- Stable source, frame, and keyframe identities.
- Canonical monotonic timestamps for session compatibility.
- Explicit frame rejection for unsupported layout, timestamp problems, resolution changes, exact duplicates, and insufficient quality.
- Sharpness, contrast, luma, clipping, and normalized quality metrics.
- Every-frame, fixed-interval, and target-FPS sampling.
- Initial keyframe selection using temporal spacing and visual change.
- Bounded decoded-keyframe cache with deterministic eviction.
- Cache policies derived from Phase 8.1 memory profiles.
- Efficient behavior for RTX 2060 6 GB and High Quality behavior for RTX 5090 32 GB.
- Structured ingestion and keyframe status/progress events with throttled progress cadence.
- Versioned JSON-compatible ingestion report metadata.
- Cooperative cancellation through `ReconstructionOperationControl`.
- Safe application of ingestion contracts to a matching empty `ReconstructionSession`.
- No Qt, PyTorch, SfM, SLAM, or meshing dependency in the new ingestion modules.

## Validation completed during package preparation

Executed successfully in the package-preparation environment:

```text
Python compilation:                       PASS
Phase 8.2 quick suite:                    58 passed
Phase 8.2 integration/stress suite:        2 passed
Phase 8.2 non-UI ingestion smoke:         PASS
Phase 8.1 contracts smoke regression:     PASS
Synthetic RTX 2060 ingestion policy:      Efficient / 48 cached frames
Synthetic RTX 5090 ingestion policy:      High Quality / 512 cached frames
```

The package-preparation environment does not contain PyQt6 or VisPy, so the complete historical UI/full application regression must be executed in the project `depth` conda environment.

## Package verification

```bash
cd ~/Downloads

sha256sum -c \
  eia_depth_phase8_2_ingestion_keyframes.zip.sha256
```

Expected:

```text
eia_depth_phase8_2_ingestion_keyframes.zip: OK
```

## Baseline verification

```bash
cd /home/eiatech/Desktop/depth_3d_model_lab
conda activate depth

git branch --show-current
git status -sb
git log -4 --oneline --decorate
```

Expected branch and latest commit:

```text
phase8-reconstruction
0a41bdc feat(reconstruction): add Phase 8 foundation contracts
```

The working tree must be clean before applying the package.

## Installation

```bash
cd /home/eiatech/Desktop/depth_3d_model_lab

unzip -o \
  ~/Downloads/eia_depth_phase8_2_ingestion_keyframes.zip \
  -d .

chmod +x scripts/run_phase8_2_tests.sh
chmod +x scripts/run_reconstruction_ingestion_smoke.py
```

## Compilation

```bash
python -m compileall -q \
  core/reconstruction \
  tests/reconstruction_ingestion_helpers.py \
  tests/unit/test_reconstruction_ingestion_*.py \
  tests/unit/test_phase8_2_backward_compatibility.py \
  tests/integration/test_reconstruction_ingestion_lifecycle.py \
  tests/stress/test_reconstruction_ingestion_stress.py \
  scripts/run_reconstruction_ingestion_smoke.py
```

No output means compilation passed.

## Focused tests

```bash
./scripts/run_phase8_2_tests.sh quick
```

Expected from the packaged implementation:

```text
58 passed
```

```bash
./scripts/run_phase8_2_tests.sh integration
```

Expected:

```text
2 passed
```

## Non-UI smoke

Synthetic sequence:

```bash
python scripts/run_reconstruction_ingestion_smoke.py
```

Real image sequence:

```bash
python scripts/run_reconstruction_ingestion_smoke.py \
  --input /path/to/image_sequence \
  --profile balanced \
  --output-dir /tmp/phase8_2_real_sequence_smoke
```

Recorded video:

```bash
python scripts/run_reconstruction_ingestion_smoke.py \
  --input /path/to/video.mp4 \
  --profile efficient \
  --output-dir /tmp/phase8_2_video_smoke
```

The smoke produces:

- `ingestion_report.json`
- `keyframes_contact_sheet.png`
- A clear terminal PASS/FAIL summary

It does not open the application UI or run SfM/SLAM.

## Full automated regression

```bash
./scripts/run_phase8_2_tests.sh full
```

This runs all unit, UI, integration, and stress tests. The two documented optional real Ultralytics/SAM hardware tests may remain skipped. No test should fail.

## Targeted manual validation

Only the following checks are required:

1. Start the application with `python main.py`.
2. Confirm the header still displays exactly `v0.7.5`.
3. Confirm Reconstruction remains disabled and no new fake controls appear.
4. Open one image and run one existing depth inference.
5. Confirm the depth map and 3D point cloud still appear.
6. Run the Phase 8.2 smoke on one small real image sequence or recorded video and inspect the keyframe contact sheet.
7. Close the application normally.

## Git review and delivery

```bash
git diff --check
git status --short
git diff --stat

git add -A

git diff --cached --check
git diff --cached --stat
git diff --cached --name-only
```

Confirm that only Phase 8.2 source, scripts, tests, and this validation document are staged. Do not stage smoke outputs, caches, logs, model weights, datasets, or temporary files.

Commit and push only after local validation passes:

```bash
git commit -m "feat(reconstruction): add validated ingestion and keyframe selection"
git push
```

Do not create a tag and do not merge into `main` during Phase 8 development.
