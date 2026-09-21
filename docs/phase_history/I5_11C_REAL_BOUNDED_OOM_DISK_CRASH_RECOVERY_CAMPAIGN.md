# I5.11C — Real bounded OOM / disk / crash / recovery campaign

I5.11C executes the three failure modes frozen by I5.11A/B exactly once each,
through the existing whole-engine process-group supervisor, followed by exactly
one clean recovery attempt after each successfully-contained expected fault.

## Frozen scenario order

1. `cuda_oom` — synthetic CUDA OOM signature; no GPU memory exhaustion.
2. `disk_enospc` — synthetic `errno.ENOSPC`; no disk filling.
3. `crash_orphan_descendant` — bounded parent crash with a live descendant.

The worker is the I5.11B-frozen `scripts/i5_11_bounded_fault_worker.py`.
I5.11C does not modify that worker, the production failure classifier, the
recovery engine, reconstruction defaults, Hybrid, or Auto policy.

## Evidence semantics

A fault attempt passes only when the frozen signature/state/exit behavior is
observed and the existing supervisor leaves the process group clean and removes
the failed owned workspace. A clean recovery must then complete in a fresh
workspace, preserve its recovery result, and leave no descendant process alive.

`robustness_failure` is completed scientific evidence and is terminal; it is not
retried and may proceed to I5.11D qualification. `runtime_integrity_failure`
means the intended controlled experiment was not completed and does not
authorize I5.11D.

## Abort rules

The campaign stops immediately on the B-frozen conditions: disk safety-floor
breach, pre-existing fault worker, repository-anchor drift, unexpected fault
signature/state/exit, process-group leak, failed workspace cleanup, or failed
clean recovery. There is no retry-until-pass.

## Claim boundary

These are synthetic robustness faults, not reconstruction-quality experiments.
No provider ranking, performance ranking, metric-scale claim, metrology claim,
production classifier change, real GPU OOM, or real disk exhaustion is allowed.
