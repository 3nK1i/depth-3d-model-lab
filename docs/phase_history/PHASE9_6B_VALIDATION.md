# Phase 9.6B — Unified Representation / Scene Composition

Phase 9.6B adds a read-only scene-composition projection over the qualified
Reconstruction presentation paths introduced through Phases 9.3–9.6A.

## Scope

- Canonical surface is the only spatial, picking, and measurement authority.
- Persisted semantic objects are supporting membership/context only.
- Persistent world/camera context is supporting context and not scale evidence.
- Gaussian and NeRF remain optional appearance-only layers.
- All composed persistent layers must match the canonical scene identity.
- Visibility never changes authority or persisted evidence.
- Missing optional layers degrade to the canonical scene without substitution.
- No persistence schema, reconstruction controller, canvas renderer, backend, measurement, or QC contract is changed.

## Validation gates

```bash
./scripts/run_phase9_6b_tests.sh quick
./scripts/run_phase9_6b_tests.sh integration
./scripts/run_phase9_6b_tests.sh backward
```

Manual validation is intentionally limited to the affected unified-workspace composition summary and existing scene-layer/representation switching behavior.
