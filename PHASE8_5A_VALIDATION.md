# Phase 8.5A — Global Consistency Foundation

Base: `d439580fb186d69567f1de26c69a2c69d77f89f1` on `phase8-reconstruction`.

## Scope

Phase 8.5A introduces dependency-light contracts for loop-closure candidates, verified pose-graph constraints, optimized keyframe poses, relocalization results, map revisions, and a deterministic mock optimizer used only for contract/integration validation.

The step deliberately does **not** enable native stella loop closure/global optimization yet. Existing `LiveSlamRequest` protection remains in place. Production/native integration belongs to Phase 8.5B.

All monocular poses remain `T_reconstruction_from_camera` in arbitrary reconstruction gauge with `unit_to_metre=None`. Raw tracking and optimized poses are represented as distinct layers.

## Validation

```bash
./scripts/run_phase8_5a_tests.sh quick
./scripts/run_phase8_5a_tests.sh integration
./scripts/run_phase8_5a_tests.sh stress
python scripts/run_global_consistency_smoke.py
```

Then run the established Phase 8.4D → 8.1 backward quick gates, isolation checks, and the full suite before commit.

Reconstruction UI must remain disabled. No dense fusion is included.

## Phase 8.5 sequencing

- **8.5A** — backend-neutral global-consistency contracts, pose-graph semantics, relocalization acceptance, deterministic mock validation.
- **8.5B** — isolated native stella loop-candidate/relocalization/global-optimization worker protocol and real backend validation.
- **8.5C** — live optimized-trajectory integration, map-revision/correction propagation, drift-correction behavior.
- **8.5D** — loop/relocalization hardening, repeated-loop stress, long-run validation, and final Phase 8.5 signoff.
