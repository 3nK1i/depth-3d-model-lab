# Phase 9.6E1 — Digital Twin Processing / Job Foundation

Baseline: `712c15919a31f7bbeda09771cc2d6bc3c0410461`

## Scope

Phase 9.6E1 adds a reusable application-side contract and Qt supervisor for future long-running Digital Twin build jobs. It deliberately does **not** add full-scene DA3 execution or a NeRF training button yet.

The foundation provides:

- one active heavyweight build job at a time;
- explicit job identity, kind, backend/runtime and output provenance;
- preparing/running/cancelling/completed/failed/cancelled lifecycle;
- finite or indeterminate progress plus stage/message/unit counts;
- elapsed-time presentation;
- shell-free `QProcess` execution;
- isolated POSIX process group/session;
- bounded `SIGTERM -> SIGKILL` cancellation and shutdown;
- fail-closed rejection of backend claims to canonical geometry, metric-scale, measurement, QC or certified-metrology authority;
- canonical workspace fallback on failure/cancellation;
- a compact collapsed `BUILD & PROCESSING` UI surface;
- adapter seams for Phase 9.6E2 full-scene DA3 and Phase 9.6E3 NeRF Build/Train/Rebuild.

## Authority / isolation

No heavy DA3, gsplat, Nerfstudio, tiny-cuda-nn or torch backend is imported by the E1 foundation. The production `depth` environment remains orchestration/presentation only.

A processing job is never canonical geometry authority, metric-scale authority, measurement authority, QC authority, or certified metrology. Dense-geometry jobs remain supporting geometry until separately registered/canonicalized; Gaussian/NeRF jobs remain appearance only.

## Deferred intentionally

- Full-scene recorded-video DA3 keyframe selection/execution: 9.6E2.
- NeRF build/train/rebuild product UI and quality profiles: 9.6E3.
- Final workspace control consolidation: 9.6E4.
- Persistence/restore of build-job state/results: 9.6F.
- Cross-backend GPU admission/resource coordination: 9.6G.

## Validation gates

```bash
./scripts/run_phase9_6e1_tests.sh quick
./scripts/run_phase9_6e1_tests.sh integration
./scripts/run_phase9_6e1_tests.sh backward
```

Target production-environment counts for this delivery:

- quick: 14 passed
- integration: 6 passed
- backward: 48 passed

No unexplained failures or skips are accepted.

## Production integration fixture correction

The production PyQt integration run exposed a timeout in the nominal successful-worker test. The root cause was the test fixture, not the supervisor lifecycle: the fixture declared `out-complete/` as its job `output_dir` but emitted `result.bin` from the parent temporary directory. The E1 supervisor correctly rejected that result as outside the declared output directory, leaving the test waiting for a `COMPLETED` state that must never be admitted for an unsafe result path.

The successful fixture now writes and reports `out-complete/result.bin`, preserving the strict result-path confinement contract. The earlier speculative fast-worker lifecycle patch is intentionally reverted; no production lifecycle semantics are changed by this correction.
