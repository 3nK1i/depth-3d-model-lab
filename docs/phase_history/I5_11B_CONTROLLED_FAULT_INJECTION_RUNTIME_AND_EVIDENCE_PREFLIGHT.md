# I5.11B — Controlled Fault-Injection Runtime / Evidence Preflight

## Purpose

I5.11B freezes the exact runtime and evidence contract for the bounded I5.11C
failure/recovery campaign admitted by I5.11A. B does **not** execute any
supervised fault worker and does not pressure GPU or disk resources.

## Frozen predecessor

- I5.11A formal admission SHA256:
  `9d1f02e4cc0cc90eb909cf17ea5621a3aaf580a75ddf2a2269ca4402d5cdf1e6`
- post-I5.10D delta SHA256:
  `773f63c31d27d5746fe788f9a0463792a6d5061676fea6f44cb004f99e4a6247`

## Bounded fault worker

B adds `scripts/i5_11_bounded_fault_worker.py` and freezes its source hash.
The worker has four deterministic modes:

- `cuda_oom`: emits `simulated cuda out-of-memory`, then raises a synthetic
  exception. It allocates no GPU memory.
- `disk_enospc`: raises `OSError(errno.ENOSPC, "No space left on device")`.
  It does not fill a filesystem.
- `crash_orphan_descendant`: exits with code 17 while a child remains alive so
  the existing process-group supervisor must reap the descendant.
- `complete`: clean recovery control; terminates its child and writes a small
  recovery result in the supervisor-owned workspace.

B may run the worker's `--help` in automated tests. It does not execute any
fault or recovery mode.

## Classifier preflight

The production failure classifier is exercised only as a pure in-process
function call:

- synthetic CUDA OOM remains first-class `cuda_out_of_memory` and requires a
  worker restart;
- synthetic ENOSPC remains the current `unknown_model_error`, while its raw
  `No space left on device` signature is preserved.

The ENOSPC classifier gap is evidence, not silently fixed in B.

## Disk safety

B reads free-space metadata only; it creates no pressure/probe files. I5.11C is
not authorized unless the selected evidence filesystem has at least 4 GiB free.
This floor is an operational abort guard, not a disk-pressure experiment.

## Frozen I5.11C order

1. `cuda_oom` fault attempt 001 → clean recovery attempt 001
2. `disk_enospc` fault attempt 001 → clean recovery attempt 001
3. `crash_orphan_descendant` fault attempt 001 → clean recovery attempt 001

Each fault and recovery is exactly once. Attempt directories are new and
immutable. There is no retry-until-pass.

## Required containment

Every fault must show the expected signature/state, leave the process group
clean, and remove the failed supervisor-owned workspace. Every recovery must
complete, leave no leaked descendants, and preserve its successful workspace.

Unexpected signatures, process leaks, cleanup failures, recovery failures,
repository-anchor drift, or disk-safety violations fail closed.

## Scope exclusions

I5.11B performs no real GPU OOM, no real disk fill, no SfM/MVS/provider run,
no production classifier modification, no recovery-engine modification, no
production-default change, no Hybrid authorization, and no Auto policy.

## Persisted JSON fault-order validation correction

The formal I5.11A artifact is serialized with sorted JSON object keys. JSON object
member order is therefore not an execution-order authority. I5.11B validates
that `fault_matrix` contains exactly the three admitted fault members, while
the frozen campaign execution order is validated from the explicit
`i5_11c_campaign_contract.fault_scenarios` list:

1. `cuda_oom`
2. `disk_enospc`
3. `crash_orphan_descendant`

This is a validation-layer correction only. It does not execute faults, alter
the production failure classifier or recovery engine, modify defaults, relax
safety constraints, or authorize real GPU/filesystem exhaustion.
