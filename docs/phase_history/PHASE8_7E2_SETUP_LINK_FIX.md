# Phase 8.7E2 Nerfstudio setup link fix

## Scope

This corrective overlay addresses the real RTX 2060 setup failure observed while building the pinned tiny-cuda-nn PyTorch bindings. The build reached its final linker step and failed with `ld: cannot find -lcuda`.

## Root cause

Two setup-script issues were found:

1. `conda run` may emit surrounding blank lines. The old `tail -n 1` extraction could therefore resolve the isolated environment prefix to an empty string, which produced `CUDA_HOME=` in the failed build command.
2. CUDA's `libcuda.so` link stub is not necessarily in the compiler's default library search path. The tiny-cuda-nn extension links with `-lcuda`, so the setup must expose the isolated toolkit's driver-stub directory at build time.

## Correction

- Resolve the isolated prefix from `sys.prefix`, stripping blank lines and failing closed if it is empty.
- Auto-discover `libcuda.so` beneath the isolated CUDA toolkit.
- Pass the stub directory only through `LIBRARY_PATH` for the tiny-cuda-nn build.
- Do not add the CUDA stub directory to `LD_LIBRARY_PATH`; runtime must continue to use the real NVIDIA driver library.
- Add `--resume` so a verified partial E2 environment/checkouts can continue from the native build instead of redownloading/recreating the environment.
- `--resume` re-verifies Python, PyTorch/CUDA, Nerfstudio revision and tiny-cuda-nn revision before continuing.

## User recovery path

Apply this overlay, rerun the focused E2 tests, then execute:

```bash
./scripts/setup_nerfstudio_backend.sh --resume
```

Do not use `--recreate` for this observed failure unless the resume path reports that the existing environment/checkouts are inconsistent.
