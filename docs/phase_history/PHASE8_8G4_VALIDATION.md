# Phase 8.8G4 — Failure / Recovery UX

## Scope

G4 adds a compact, generic failure/recovery presentation to the existing Reconstruction session panel. It does not add another recovery engine.

- Existing live-SLAM bounded worker recovery remains authoritative and is presented while `RECOVERING`.
- Terminal failure/cancellation may expose **Retry From Source** only when the controller is idle, the selected source is ready, and no retained scene/checkpoint evidence makes a fresh session ambiguous.
- Retry delegates to the existing Reconstruction start path and creates a fresh attempt. It is not checkpoint resume and does not manually restart an in-flight live worker.
- Preserved scene/checkpoint evidence is surfaced instead of silently discarded. Compatible/prepared checkpoint continuation remains owned by G3.
- Session-identity conflicts fail closed.
- No persistence/checkpoint schema changes, heavy-backend imports, domain-specific branching, authority promotion, or metrology certification are introduced.

## Validation

Run:

```bash
./scripts/run_phase8_8g4_tests.sh quick
./scripts/run_phase8_8g4_tests.sh protocol
./scripts/run_phase8_8g4_tests.sh backward
python scripts/run_phase8_8g4_failure_recovery_smoke.py
python scripts/run_phase8_8g4_failure_recovery_preview.py
```

The smoke must prove bounded live recovery is observation-only, source retry is explicit/fresh, retained evidence blocks ambiguous retry, checkpoint execution is not performed by G4, and authority remains unchanged.
