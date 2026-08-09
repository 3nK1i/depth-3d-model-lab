# Phase 8.7E2 — Real Nerfstudio / Nerfacto Backend Qualification

## Scope

Phase 8.7E2 qualifies one real isolated NeRF backend: Nerfstudio `nerfacto`. E2 proves that the pinned backend can be installed outside the production `depth` environment, execute bounded real CUDA training on the target RTX 2060 profile, save a reloadable config/checkpoint, and produce a real rendered RGB artifact.

E2 is **not** a real-scene quality benchmark or final NeRF performance benchmark. Those remain Phase 8.7E3 and 8.7E4.

## Reproducible isolated runtime

The E2 setup script creates a dedicated environment and external source checkouts:

- Conda environment: `eia-nerf-nerfstudio`
- External root: `~/Desktop/eia-nerf-backends`
- Python: `3.10`
- PyTorch: `2.1.2+cu118`
- torchvision: `0.16.2+cu118`
- CUDA toolkit: `11.8.0`
- Nerfstudio: `v1.1.5` (`6b60855` revision prefix)
- tiny-cuda-nn: `b3473c81396fe927293bdfd5a6be32df8769927c`
- RTX 2060 tiny-cuda-nn target architecture: `75`

The production `depth` environment is not modified.

## Geometry / camera convention

EIA reconstruction uses OpenCV/COLMAP-style camera axes (X right, Y down, Z forward) with camera-to-reconstruction transforms named `T_reconstruction_from_camera`.

The Nerfstudio dataset adapter preserves the camera-to-world meaning while converting the camera basis expected by Nerfstudio by negating camera Y/Z columns. The adapter also disables automatic pose orientation/centering/scaling for E2 and writes explicit train/validation/test filenames.

This conversion is covered by focused tests and the dependency-light E2 contract smoke.

## Qualification scene and claims

E2 uses a deterministic six-view, small-resolution known-pose synthetic scene only to exercise the real backend path. The scene is deliberately bounded and arbitrary-scale.

A passing E2 run proves:

- isolated Nerfstudio/tiny-cuda-nn runtime is available;
- pinned source revisions match;
- real CUDA is available on the expected RTX 2060 profile;
- bounded Nerfacto training completes without OOM;
- Nerfstudio writes a config and checkpoint;
- the checkpoint can be loaded for real test-view rendering;
- a non-empty RGB preview is produced;
- the resulting NeRF artifact remains appearance-only.

It does **not** claim production reconstruction quality, PSNR/SSIM/LPIPS ranking, production VRAM/FPS, or certified geometry/metrology.

## Dependency-light validation

Run before creating the isolated runtime:

```bash
./scripts/run_phase8_7e2_tests.sh quick
./scripts/run_phase8_7e2_tests.sh integration
./scripts/run_phase8_7e2_tests.sh stress
python scripts/run_nerf_e2_contracts_smoke.py
```

## Isolated backend setup

Only after the source/test gates are green:

```bash
./scripts/setup_nerfstudio_backend.sh
python scripts/run_nerf_environment_probe.py
```

The setup script refuses to overwrite an existing E2 environment/checkouts unless `--recreate` is explicitly provided.

## Real RTX 2060 qualification

```bash
rm -rf /tmp/phase8_7e2_nerf_qualification
python scripts/run_nerf_backend_qualification.py \
  --hardware-profile rtx_2060_6gb \
  --training-steps 16
```

A passing summary must record:

- `pass=true`
- `real_backend_qualified=true`
- `training_completed_without_oom=true`
- real RTX 2060 hardware evidence
- expected Nerfstudio and tiny-cuda-nn revisions
- valid config/checkpoint/preview paths
- `scene_role=appearance_only`
- `canonical_geometry_authoritative=false`
- `certified_metrology=false`
- `reconstruction_ui_enabled=false`

## Backward/isolation gates

- E1 contracts remain valid.
- D3/D2 Gaussian quick gates remain valid.
- `pyproject.toml` remains free of Nerfstudio/tiny-cuda-nn production dependencies.
- `core/reconstruction` contains no direct `nerfstudio` or `tinycudann` imports.
- real backend imports remain inside the isolated worker.

## Explicit deferrals

Phase 8.7E2 does not own:

- real-scene/held-out NeRF quality evaluation (E3);
- production NeRF VRAM/training/render benchmarking (E4);
- NeRF artifact/lifecycle hardening beyond E2 qualification (E4);
- whole reconstruction-engine recovery (8.7G);
- Gaussian/NeRF integration into `PersistentReconstructionScene` (8.7F);
- Reconstruction UI (8.8).
