# Phase 9.4C — Real Gaussian Backend Integration / Optimization Execution

Baseline: Phase 9.4B clean closure (`43a0001e9eae13a18dfc56b5adc07ea8a0e6c81e`).

## Scope

Phase 9.4C connects the qualified Phase 9.4A external `gsplat` runtime and the
Phase 9.4B canonical `NeuralSceneDataset` to the already-qualified Phase 8 real
`gsplat_worker.py`.

It does **not** add application UI, persistence of Gaussian representations,
interactive Gaussian rendering, NeRF fallback, Graphdeco production use, or
metric/measurement/QC authority.

## Locked authority boundary

- persistent canonical scene: canonical geometry authority;
- Gaussian artifact: appearance only;
- Gaussian metric-scale authority: false;
- Gaussian measurement authority: false;
- Gaussian QC authority: false;
- certified metrology: false.

An arbitrary-scale canonical scene remains arbitrary scale. Optimization never
creates `unit_to_metre`.

## Input chain

1. exact Phase 9.4B `NeuralSceneDataset` manifest;
2. integrity-verified persistent canonical OBJ;
3. deterministic bounded Gaussian seed whose positions are exact canonical mesh
   vertices;
4. READY Phase 9.4A runtime report for the exact clean qualified gsplat checkout.

No image/pose rebinding, DA3 rerun, or geometry reconstruction occurs here.

## Runtime lifecycle

The heavy worker is started only through the isolated
`eia-gaussian-gsplat` interpreter. The production `depth` process does not
import `gsplat`. The child uses:

- `shell=False`;
- `start_new_session=True`;
- bounded timeout;
- SIGTERM then SIGKILL escalation;
- child-only conda/CUDA environment variables;
- no mutation of the external checkout or environment.

## Real qualification target

The target-machine command uses the Phase 9.3J South Building persistent scene
already materialized by 9.4B, defaults to the efficient profile, and requires:

- real CUDA execution;
- completed optimization;
- positive measured VRAM;
- non-empty `scene.pt`;
- non-empty optimized preview;
- exact dataset fingerprint match;
- canonical seed provenance;
- appearance-only authority;
- zero heavy Gaussian importability in `depth` before/after execution.

Quality benchmarking, held-out metrics, interactive viewer integration, and
long-run resource hardening remain later 9.4 substeps.
