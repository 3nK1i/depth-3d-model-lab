# Phase 8.7F1 — Reconstruction Pipeline Integration Contracts + Orchestration Foundation

Required baseline:

- branch: `phase8-reconstruction`
- HEAD: `d9475753a86f896fa3b29d1e4617887b27d7f26b`
- working tree clean and synchronized with `origin/phase8-reconstruction`

## Scope

Phase 8.7F1 creates the dependency-light orchestration layer that later F2-F5 steps bind to already-qualified reconstruction components. It deliberately does **not** execute DA3, canonical geometry, Gaussian, Nerfstudio, PyCOLMAP, stella_vslam, or UI code.

The foundation adds:

- a stable reconstruction pipeline stage DAG;
- high-level requested output goals;
- generic provider roles and multi-role provider descriptors;
- deterministic provider priority, availability and fallback resolution;
- required vs optional/non-authoritative stage semantics;
- generic artifact kinds with immutable authority rules;
- explicit canonical-geometry vs semantic/QC vs appearance-only separation;
- deterministic sequential orchestration state transitions;
- cooperative cancellation checkpoints;
- provider failure capture and bounded fallback;
- explicit degraded/failed/cancelled pipeline results;
- provider provenance aggregation into existing `BackendProvenance`;
- JSON plan/result evidence;
- deterministic development-only mock executors for validation.

## Authority invariants

- Dense geometry and surface mesh are the only F1 artifact classes marked `canonical_geometry`.
- Depth and camera pose remain supporting geometry/evidence.
- Semantic geometry remains semantic context.
- Inspection/QC remains an inspection decision layer.
- Gaussian and NeRF outputs are permanently `appearance_only` in the F1 stage contract.
- F1 cannot relabel a Gaussian or NeRF artifact as canonical geometry.
- `certified_metrology=false` remains mandatory.
- Reconstruction UI remains disabled until Phase 8.8.
- Development mock execution is explicitly recorded and cannot claim production execution.

## Provider/fallback semantics

Providers are ordered by `(priority, provider_id)`. The plan records both available and unavailable candidates. At runtime, one provider may fail and the orchestrator may try the next declared candidate.

- A fallback that succeeds is recorded explicitly and makes the run `DEGRADED` rather than hiding the substitution.
- Required canonical/semantic/QC stages fail closed if no provider succeeds.
- Gaussian and NeRF stages may be skipped only when optional degradation is allowed, because they are non-authoritative appearance outputs.
- If optional degradation is disabled, requested appearance stages become required.

## Deliberate future ownership

F1 defines slots/contracts only. The following remain untouched:

- **8.7F2** — Depth Anything 3 isolated qualification + depth/pose provider binding.
- **8.7F3** — real canonical geometry + semantics + QC component integration.
- **8.7F4** — real Gaussian + NeRF appearance integration.
- **8.7F5** — `PersistentReconstructionScene`, lifecycle/provenance/fallback and final F validation.
- **8.7G** — whole-engine cancellation/recovery/checkpoint/endurance hardening.
- **8.8** — Reconstruction application/controller/UI.

There is no `PersistentReconstructionScene` implementation in F1.

## Focused validation

```bash
./scripts/run_phase8_7f1_tests.sh quick
./scripts/run_phase8_7f1_tests.sh integration
./scripts/run_phase8_7f1_tests.sh stress
python scripts/run_reconstruction_orchestration_smoke.py
```

The smoke intentionally marks one preferred depth provider unavailable so the fallback path is observable. A successful smoke remains `DEGRADED` by design while preserving canonical geometry authority and appearance-only neural outputs.

Artifacts:

```text
/tmp/phase8_7f1_orchestration/plan.json
/tmp/phase8_7f1_orchestration/result.json
/tmp/phase8_7f1_orchestration/summary.json
```

## Backward-safety expectations

- Phase 8.7A-E schema versions remain unchanged.
- `InferenceController` remains model-agnostic and is not coupled to F1.
- `pyproject.toml` gains no DA3/Nerfstudio/tiny-cuda-nn dependency.
- Production reconstruction core does not directly import DA3, Nerfstudio, tiny-cuda-nn, gsplat or torch for F1.
- Existing Gaussian and NeRF artifact contracts remain separate.
- No UI flag is enabled.

## Full regression

F1 is a bounded foundation step, so full regression is not required before the initial focused commit unless a focused/backward gate exposes unexpected cross-cutting behavior. Full Phase 8.7 regression remains mandatory at 8.7G4.
