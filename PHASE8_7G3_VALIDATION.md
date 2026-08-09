# Phase 8.7G3 — Endurance / Failure / Degradation / Backward Compatibility

## Scope

G3 proves that the integrated reconstruction engine remains stable across repeated
whole-engine interruption/recovery cycles and controlled degradation/failure
conditions. It composes the existing G1 lifecycle containment and G2 checkpoint
resume contracts; it does not reimplement them.

Full Phase 8.7 regression and the final UI-readiness decision remain owned by G4.
Reconstruction UI remains disabled until Phase 8.8.

## Baseline

- Branch: `phase8-reconstruction`
- Required starting commit: `a84bf15fe6ee49ff91a9f0989efc09c59a470abf`
- G1 whole-engine lifecycle containment remains authoritative.
- G2 checkpoint/resume remains authoritative.

## Required invariants

- Repeated cancelled engine attempts leave no process group and no failed workspace.
- Each clean recovery produces a fresh persistent-scene identity; stale scene reuse is rejected.
- Canonical geometry remains authoritative across clean/degraded operation.
- Gaussian and NeRF remain `appearance_only` and can never become geometry authority.
- Optional appearance failure is explicit `degraded`, not silently `completed`.
- Required canonical-geometry failure fails closed and cannot publish a persistent scene.
- Incompatible/provider-drifted checkpoints are rejected safely.
- Checkpoint/resume remains functional after the endurance campaign.
- Certified metrology remains false; reconstruction UI remains disabled.

## Focused validation

```bash
./scripts/run_phase8_7g3_tests.sh quick
./scripts/run_phase8_7g3_tests.sh integration
./scripts/run_phase8_7g3_tests.sh stress
python scripts/run_reconstruction_g3_contracts_smoke.py
```

## Real bounded campaign

```bash
rm -rf /tmp/phase8_7g3_real_endurance

python scripts/run_reconstruction_g3_real_endurance.py \
  --output-dir /tmp/phase8_7g3_real_endurance \
  --cycles 2 \
  --cancel-after-s 1.5 \
  --fault-nerf-training-steps 1000 \
  --recovery-nerf-training-steps 16
```

Each real cycle cancels an active integrated F4→F5 process tree, verifies cleanup,
then performs a fresh bounded real Gaussian+Nerfacto recovery and persistent-scene
write. After the repeated cycles, G3 executes controlled optional degradation,
required-stage fail-closed, checkpoint incompatibility rejection, then runs the real
G2 process-restart checkpoint/resume gate once more to prove cross-feature recovery.

This is not another Gaussian/NeRF quality or performance qualification. D3/E3/E4
remain authoritative for those backend-specific concerns.

## Expected real evidence

- at least 2 repeated real cancel/recovery cycles
- all cancelled attempts `safe_for_recovery=true`
- all process groups clean
- all failed workspaces removed
- all recovery pipelines `completed`
- all recovery persistent scenes `ready`
- unique scene IDs across clean cycles
- `stale_state_detected=false`
- optional degradation PASS
- required failure fail-closed PASS
- checkpoint incompatibility rejection PASS
- post-endurance G2 process-restart resume PASS
- canonical geometry authoritative
- Gaussian/NeRF `appearance_only`
- certified metrology false
- full regression deferred to G4
- reconstruction UI disabled

## Deferred ownership

- G4: complete Phase 8.7 regression, final backward-compatibility gate, UI-readiness decision
- 8.8: reconstruction application/UI
- 8.9: production release validation/profiling
