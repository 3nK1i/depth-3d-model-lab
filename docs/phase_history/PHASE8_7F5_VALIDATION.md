# Phase 8.7F5 — PersistentReconstructionScene + final F validation

## Scope

F5 closes Phase 8.7F integration by materializing the F1–F4 orchestration result as a durable, reloadable `PersistentReconstructionScene` manifest.

The scene records:

- stable scene identity and SHA-256 content fingerprint;
- canonical geometry artifact authority;
- semantic and inspection/QC artifacts;
- Gaussian and NeRF appearance-only artifacts;
- provider provenance;
- explicit fallback/degradation evidence;
- file-backed neural artifact integrity references when available;
- manifest checksum and reload verification.

## Authority boundary

- canonical dense/surface geometry remains authoritative;
- semantics remain semantic context;
- inspection/QC remains inspection authority;
- Gaussian and NeRF remain `appearance_only`;
- F5 never enables certified metrology;
- reconstruction UI remains disabled until Phase 8.8.

## Deferred responsibilities

F5 does **not** implement whole-engine cancellation/resource cleanup, engine checkpoint/resume, endurance campaigns, or UI. Those remain Phase 8.7G and 8.8 responsibilities.

## Validation gates

```bash
./scripts/run_phase8_7f5_tests.sh quick
./scripts/run_phase8_7f5_tests.sh integration
./scripts/run_phase8_7f5_tests.sh stress
python scripts/run_reconstruction_f5_contracts_smoke.py
```

Real persistence reuses the already-qualified F4 result and does not rerun Gaussian or Nerfacto:

```bash
python scripts/run_reconstruction_f5_real_persistence.py \
  --f4-result /tmp/phase8_7f4_real_backends/pipeline_result.json \
  --f4-summary /tmp/phase8_7f4_real_backends/summary.json \
  --output-dir /tmp/phase8_7f5_real_persistence
```

Expected invariants:

- scene state `ready` for a complete F4 result;
- stable `prs-...` scene ID and full SHA-256 fingerprint;
- canonical geometry authoritative;
- Gaussian/NeRF appearance-only;
- certified metrology false;
- manifest reload passes checksum + normalized fingerprint validation;
- referenced file integrity passes;
- whole-engine hardening deferred to 8.7G;
- reconstruction UI disabled.
