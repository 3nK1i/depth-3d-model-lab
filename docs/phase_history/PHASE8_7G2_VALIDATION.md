# Phase 8.7G2 — Whole-Engine Checkpoint + Resume

## Scope

G2 adds durable, backend-neutral checkpoint/resume semantics to the integrated
F1–F5 reconstruction pipeline. It does not implement the endurance/failure
campaign (G3), the full Phase 8.7 regression/UI-readiness gate (G4), or UI work.

## Baseline

- Branch: `phase8-reconstruction`
- Required starting commit: `a0343f3e9ba9fb3a62bb1077214ad8231067b7da`
- G1 whole-engine cancellation/recovery/resource cleanup remains authoritative.

## Contracts

- Checkpoints are atomic JSON manifests with SHA-256 sidecars.
- Checkpoints carry a UUID identity plus a deterministic semantic fingerprint.
- Request, plan, provider version/adapter/execution identity must match at resume.
- Only a contiguous completed/skipped stage prefix can be reused.
- Required stages may never be restored as skipped.
- Every committed artifact payload reference must be rehydrated before resume.
- Completed stages are seeded into the orchestrator and are not re-executed.
- Canonical geometry authority, semantic/QC authority and appearance-only neural
  authority remain unchanged after resume.
- Certified metrology remains false and reconstruction UI remains disabled.

## Focused validation

```bash
./scripts/run_phase8_7g2_tests.sh quick
./scripts/run_phase8_7g2_tests.sh integration
./scripts/run_phase8_7g2_tests.sh stress
python scripts/run_reconstruction_g2_contracts_smoke.py
```

Expected development counts:

- quick: 16 passed
- integration: 1 passed
- stress: 2 passed
- contract smoke: PASS

## Real process-restart validation

```bash
rm -rf /tmp/phase8_7g2_real_checkpoint_resume

python scripts/run_reconstruction_g2_real_checkpoint_resume.py \
  --output-dir /tmp/phase8_7g2_real_checkpoint_resume \
  --nerf-training-steps 16
```

The first process executes source ingestion, camera pose and depth evidence,
atomically commits a durable checkpoint after `depth_evidence`, then ends the
attempt. A second process reloads the checkpoint, validates all compatibility
constraints, blocks re-execution of the committed provider, resumes at
`dense_geometry`, executes the remaining canonical/semantic/QC stages plus the
already-qualified Gaussian and Nerfacto appearance backends, and writes/reloads
a `PersistentReconstructionScene`.

Required evidence includes:

- `process_restart_exercised=true`
- checkpoint stage `depth_evidence`
- next/first resumed stage `dense_geometry`
- three committed stages reused
- completed-stage re-execution blocked
- manifest SHA-256 verification
- provider identity/version validation
- payload reference rehydration validation
- final pipeline `completed`
- persistent scene `ready`
- canonical geometry authoritative
- Gaussian/NeRF `appearance_only`
- certified metrology false

## Deferred ownership

- G3: endurance, repeated failure/degradation campaign, backward compatibility stress
- G4: complete Phase 8.7 regression and UI-readiness gate
- 8.8: reconstruction application/UI
