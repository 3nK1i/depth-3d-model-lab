# Phase 8.7E2 — Nerfstudio Build-Isolation Fix

## Symptom

The isolated runtime and pinned tiny-cuda-nn build completed successfully, but the subsequent Nerfstudio install failed while preparing the `fpsample` source dependency with:

```text
BackendUnavailable: Cannot import 'scikit_build_core.build'
```

## Root cause

The setup script incorrectly propagated `--no-build-isolation` from the native tiny-cuda-nn build to the Nerfstudio installation. `fpsample` 1.0.x declares `scikit-build-core` and `pybind11` as PEP 517 build-system requirements. With build isolation disabled, pip cannot provision those temporary build requirements.

## Correction

- Keep `--no-build-isolation` only for tiny-cuda-nn, which must compile against the already-pinned PyTorch/CUDA environment.
- Use normal PEP 517 build isolation for the pinned Nerfstudio source install so transitive source packages receive their declared build-system dependencies.
- On `--resume`, skip rebuilding tiny-cuda-nn when it already imports successfully.
- No packages are added to the production `depth` environment.

## Resume

Apply this correction on top of the current uncommitted E2 overlay and run:

```bash
./scripts/setup_nerfstudio_backend.sh --resume
```

Do not use `--recreate` unless a later validated correction explicitly requires it.
