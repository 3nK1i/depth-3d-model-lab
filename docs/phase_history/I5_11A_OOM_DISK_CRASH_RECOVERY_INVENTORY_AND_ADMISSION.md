# I5.11A — OOM / Disk / Crash / Recovery Inventory + Admission

## Scope

I5.11A is an execution-free inventory and admission step. It inspects the current reconstruction lifecycle and recovery architecture and freezes the smallest safe I5.11 campaign. It does not inject faults, execute SfM/MVS/providers, change production defaults, add a recovery engine, or implement Hybrid/Auto.

## Existing architecture carried forward

The repository already provides a whole-engine process-group supervisor (`core/reconstruction/engine_hardening.py`) with SIGTERM/SIGKILL containment, failed owned-workspace cleanup, orphan-descendant cleanup, and explicit `safe_for_recovery` evidence. Historical Phase 8.7G1/G3 and 8.9E evidence already cover cancellation, timeout, parent crash, repeated recovery, resource release, and recovery reuse.

`core/runtime/failures.py` has first-class CUDA OOM, worker-timeout, and worker-crash categories. It does **not** currently have a first-class ENOSPC/disk-full category. I5.11A records that as an observed classification gap; it does not silently change production classification.

## Admitted bounded campaign

I5.11C may run exactly three fault scenarios, each exactly once, with one fresh clean recovery after each fault:

1. `cuda_oom` — synthetic `simulated cuda out-of-memory` signature in an isolated fault worker. No deliberate VRAM exhaustion.
2. `disk_enospc` — synthetic `OSError(errno.ENOSPC, "No space left on device")` in a supervisor-owned workspace. No filling or pressuring the real root filesystem.
3. `crash_orphan_descendant` — existing G1 parent-crash-with-live-descendant control.

Every admitted fault must preserve process-group cleanup, failed-workspace cleanup, immutable evidence, and a fresh clean recovery. A failed recovery is a robustness failure. An unexpected infrastructure/source/worker-integrity failure means the experiment is incomplete and must not be interpreted as reconstruction quality.

## Explicit exclusions

- no real GPU-memory exhaustion;
- no real root-filesystem fill or free-space pressure;
- no user/project/evidence corruption;
- no checkpoint corruption campaign;
- no cancellation/timeout rerun (historically qualified context only);
- no retry-until-pass, result-driven retuning, production-default change, provider ranking, metric/metrology claim, Hybrid, or Auto.

## Roadmap guard

A passing I5.11A authorizes only I5.11B controlled fault-injection runtime/evidence preflight. I5.11C is not authorized until B freezes the mechanisms and evidence contract. I5.12–I5.14 remain deferred.
