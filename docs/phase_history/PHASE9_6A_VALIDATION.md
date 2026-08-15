# Phase 9.6A — Unified Workspace Foundation & Authority Model

Baseline: clean synchronized `phase9-digital-twin` at `b56ae41652d55aa374717a624a968e082764525a` (Phase 9.5H closed).

## Scope

Phase 9.6A adds a dependency-light, presentation-only Digital Twin workspace composition model and a compact passive summary inside the existing Reconstruction sidebar.

It deliberately does **not** introduce a second reconstruction controller, a new persistence/session schema, a new canvas, a new measurement/QC algorithm, or a neural backend dependency in the production process.

## Authority invariants

- canonical geometry remains spatial, picking and measurement authority;
- semantics remain supporting context only;
- Gaussian and NeRF remain optional `appearance_only` representations;
- QC/reference remains a separate evidence domain;
- calibration does not imply metric scale;
- arbitrary-scale scenes remain arbitrary until qualified scale evidence exists;
- `certified_metrology=false`;
- missing Gaussian/NeRF representations do not prevent canonical-scene use;
- no cross-representation fallback can create authority.

## Automated gates

```bash
./scripts/run_phase9_6a_tests.sh quick
./scripts/run_phase9_6a_tests.sh integration
./scripts/run_phase9_6a_tests.sh backward
```

## Manual validation

Load one existing canonical scene, then one scene with Gaussian and/or NeRF evidence if available. Verify the new **Digital Twin Workspace** summary tracks the active view and available composition while the existing canonical, scene-layer, measurement, inspection/QC, Gaussian and NeRF controls remain unchanged.

For an arbitrary-scale scene, the workspace Scale row must remain unqualified/arbitrary and Safety must remain `metrology not certified`. Hiding or failing a neural appearance representation must leave canonical geometry usable.

## Deferred to later 9.6 substeps

9.6A does not consolidate or remove older controls. Unified representation composition, cross-representation navigation synchronization, interaction consolidation, broader UX consolidation, session/export consistency, coordinated heavyweight resource policy, and final real-scene qualification remain 9.6B–9.6H work.
