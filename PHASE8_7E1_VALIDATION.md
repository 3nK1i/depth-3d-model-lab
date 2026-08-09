# Phase 8.7E1 — NeRF Contracts + External Isolation

## Scope

Phase 8.7E1 introduces the backend-neutral NeRF contract and the isolated-process boundary only. It does **not** install, train, benchmark, or qualify a real NeRF runtime; those responsibilities begin in Phase 8.7E2.

## Backend policy

- Production candidate: Nerfstudio `nerfacto` external backend.
- Reproducibility target for E2 qualification: Nerfstudio `v1.1.5` (`6b60855` revision prefix).
- Nerfstudio backend license recorded as Apache-2.0.
- tiny-cuda-nn runtime dependency license recorded as BSD-3-Clause.
- Heavy runtime dependencies remain outside the production `depth` environment.
- Real NeRF execution must use an isolated Python interpreter by default.

## Geometry / metrology authority

NeRF remains an appearance representation:

- `scene_role=appearance_only`
- `canonical_geometry_authoritative=false`
- `certified_metrology=false`
- pose convention remains `T_reconstruction_from_camera`
- metric scale context may be carried as provenance, but NeRF does not become the canonical measurement surface.

## Added contracts

- `NerfBackendKind`
- `NerfOptimizationProfile`
- `NerfBackendProvenance`
- `NerfBackendRuntimePin`
- `NerfWorkerRequest`
- `NerfSceneArtifact`
- `NerfWorkerProcessSpec`
- `NerfEnvironmentProbe`
- `run_nerf_external_worker`
- `probe_nerf_environment`

The existing `NeuralSceneDataset` is reused rather than duplicated.

## Focused validation

```bash
./scripts/run_phase8_7e1_tests.sh quick
./scripts/run_phase8_7e1_tests.sh integration
./scripts/run_phase8_7e1_tests.sh stress
python scripts/run_nerf_contracts_smoke.py
```

## Backward-compatibility safeguards

- Existing Phase 8.7A/B/C/D schema versions are unchanged.
- Gaussian contracts are not rewritten into NeRF contracts.
- No `nerfstudio` / `tiny-cuda-nn` dependency is added to `pyproject.toml`.
- Production reconstruction core does not directly import Nerfstudio or tiny-cuda-nn.
- Reconstruction UI remains disabled.

## Explicit deferrals

Phase 8.7E1 does not claim:

- real Nerfstudio installation;
- real CUDA NeRF training;
- real render output;
- real backend qualification;
- PSNR/SSIM/LPIPS comparison;
- VRAM/training/render performance;
- backend crash/cancel/recovery hardening;
- UI integration.

Those are owned by E2–E4 and later 8.7F/8.7G/8.8 as defined in the locked roadmap.
