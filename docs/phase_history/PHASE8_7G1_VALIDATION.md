# Phase 8.7G1 — Whole-Engine Cancellation, Recovery & Resource Cleanup

## Scope

Phase 8.7G1 hardens the complete reconstruction-engine lifecycle boundary after
Phase 8.7F integration. It adds cooperative provider cancellation propagation,
hard process-group containment for the integrated engine, failed-workspace
cleanup, residual-child cleanup, timeout/crash containment, and explicit clean
recovery evidence.

This phase does **not** implement checkpoint/resume (8.7G2), endurance and broad
degradation campaigns (8.7G3), the complete Phase 8.7 regression/UI-readiness
gate (8.7G4), or reconstruction UI (8.8).

## Authority invariants

- Canonical geometry remains authoritative.
- Gaussian remains `appearance_only`.
- NeRF remains `appearance_only`.
- Phase 8.7G1 does not certify metrology.
- Reconstruction UI remains disabled.

## Automated validation

```bash
./scripts/run_phase8_7g1_tests.sh quick
./scripts/run_phase8_7g1_tests.sh integration
./scripts/run_phase8_7g1_tests.sh stress
python scripts/run_reconstruction_g1_contracts_smoke.py
```

Expected development baseline:

- quick: 22 passed
- integration: 2 passed
- stress: 2 passed
- lifecycle contract smoke: PASS

Backward sanity:

```bash
./scripts/run_phase8_7f5_tests.sh quick
./scripts/run_phase8_7f4_tests.sh quick
./scripts/run_phase8_7f3_tests.sh quick
./scripts/run_phase8_7f2_tests.sh quick
./scripts/run_phase8_7f1_tests.sh quick
./scripts/run_phase8_7e4_tests.sh quick
```

Expected: `10 / 8 / 16 / 36 / 32 / 77` passed.

## Real integrated gate

```bash
rm -rf /tmp/phase8_7g1_real_hardening
python scripts/run_reconstruction_g1_real_hardening.py \
  --output-dir /tmp/phase8_7g1_real_hardening \
  --cancel-after-s 1.5 \
  --fault-nerf-training-steps 1000 \
  --recovery-nerf-training-steps 16
```

The real gate deliberately cancels one F4→F5 integrated engine attempt, verifies
process-group and partial-workspace cleanup, runs residual-child crash/timeout
containment, then performs one clean real recovery using the already-qualified
Gaussian and Nerfacto runtimes. It does not requalify backend quality/performance.

Required final evidence:

- real whole-engine attempt reaches `cancelled`;
- nested integrated-engine child was active before cancellation;
- no process group remains;
- cancelled/failed partial workspaces are removed;
- residual child after a crashing engine parent is reaped;
- timeout path is contained and clean;
- clean F4→F5 recovery reaches `completed` / persistent scene `ready`;
- canonical geometry authority remains true;
- Gaussian and NeRF remain `appearance_only`;
- certified metrology remains false;
- checkpoint/resume remains deferred to 8.7G2;
- endurance remains deferred to 8.7G3;
- full regression/UI-readiness remains deferred to 8.7G4.
