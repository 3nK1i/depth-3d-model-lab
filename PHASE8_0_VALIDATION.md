# Phase 8.0 Validation — v0.7.5 Metadata Alignment

## Purpose

Phase 8.0 aligns the live repository metadata with the released Phase 7.5 baseline before reconstruction development begins.

This step does not add reconstruction code, enable Reconstruction mode, alter the runtime state machine, or modify the v0.7.5 tag.

## Baseline

- Development branch: `phase8-reconstruction`
- Baseline commit: `520bda3`
- Existing release tag: `v0.7.5`
- Release identity: `0.7.5 — Phase 7.5 Stable`

## Changes

- Normalized `core.version.__version__` to `0.7.5` without a leading `v`.
- Kept the UI responsible for rendering the display form `v0.7.5`.
- Aligned `pyproject.toml` and `config/app.yaml` with `0.7.5`.
- Aligned the configured release name with `Phase 7.5 Stable`.
- Aligned the release phase value with `7.5`.
- Updated active release-identity assertions in unit and UI tests.
- Updated the README current-status section from release-candidate wording to the stable Phase 7.5 baseline.
- Preserved historical Phase 7 and Phase 7.5 validation documents unchanged.

## Safety boundaries

- `main` must remain untouched.
- Commit `520bda3` must not be amended or rewritten.
- Tag `v0.7.5` must not be moved, recreated, or deleted.
- Reconstruction mode must remain disabled.
- No Phase 8.1 reconstruction contracts are included in this package.

## Focused validation

```bash
python -m compileall -q app core tests

python -m pytest -q \
  tests/unit/test_phase75_step1_backward_compatibility.py \
  tests/unit/test_phase75_step2_backward_compatibility.py \
  tests/unit/test_phase75_step3_backward_compatibility.py \
  tests/unit/test_phase75_step4_backward_compatibility.py \
  tests/unit/test_phase75_step5_backward_compatibility.py \
  tests/unit/test_phase75_step6_backward_compatibility.py \
  tests/unit/test_phase75_step7_backward_compatibility.py \
  tests/unit/test_phase75_step8_backward_compatibility.py \
  tests/unit/test_phase75_step10_backward_compatibility.py \
  tests/unit/test_phase6_release_metadata.py \
  tests/unit/test_phase7_release_metadata.py \
  tests/unit/test_release_metadata.py \
  tests/ui/test_phase7_release_modes.py \
  tests/ui/test_release_header.py
```

## Metadata sanity check

```bash
python - <<'PY'
from app.widgets.header_widget import APP_VERSION
from core.version import RELEASE_NAME, RELEASE_PHASE, __version__

assert __version__ == "0.7.5"
assert APP_VERSION == "v0.7.5"
assert RELEASE_NAME == "Phase 7.5 Stable"
assert RELEASE_PHASE == 7.5
print("PASS: v0.7.5 metadata is aligned")
PY
```

## Targeted manual validation

1. Start the application normally.
2. Confirm the header displays exactly `v0.7.5`, not `vv0.7.5`.
3. Confirm Measure & Inspect remains available.
4. Confirm Reconstruction remains disabled.
5. Open one image and perform one small depth/measurement sanity check.

## Git review and commit

```bash
git diff --check
git status --short
git diff --stat

git add -A
git diff --cached --check
git diff --cached --stat
git diff --cached --name-only

git commit -m "fix(release): align v0.7.5 metadata"
git push
```

Do not create a new tag for Phase 8.0.
