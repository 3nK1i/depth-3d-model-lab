# Phase 9.5B — Canonical Scene → NeRF Input Bridge

## Scope

Phase 9.5B binds already-qualified persistent canonical cameras to exact source RGB evidence and materializes the existing backend-neutral `NeuralSceneDataset` for NeRF. It also exports Nerfstudio `transforms.json` using the existing OpenCV-c2w → OpenGL-c2w convention conversion.

No Nerfstudio runtime is probed, no NeRF backend starts, and no training/rendering/persistence occurs in this phase.

## Locked authority

- Canonical scene geometry remains authoritative.
- NeRF input/appearance is never geometry, metric-scale, measurement, QC, or certified-metrology authority.
- Arbitrary-scale scenes keep `unit_to_metre = null`.
- Exact source pixel fingerprints must match persisted canonical camera bindings.
- Retained DA3 intrinsics and canonical poses are reused; DA3 is not rerun.
- NeRF does not route through or fall back to the Gaussian bridge.

## Validation

- Focused unit and integration tests cover exact fingerprint coverage, source drift rejection, pose drift rejection, retained DA3 intrinsics, Nerfstudio pose conversion, worker-contract compatibility, and heavy-module isolation.
- `scripts/run_phase9_5b_real_scene_bridge.py` validates the real Phase 9.3J South Building persistent scene without starting Nerfstudio.
