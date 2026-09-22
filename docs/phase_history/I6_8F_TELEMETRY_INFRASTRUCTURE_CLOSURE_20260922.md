# I6.8F — Job telemetry: consolidated qualification and scoped closure

Date: 2026-09-22  
Repository: `depth_3d_model_lab`  
Branch: `hotfix/v0.9.0-video-calibration-context` (Gitea)  
Evidence baseline: committed and pushed `6576a2f` (`test(reconstruction): qualify I6.8F dense telemetry integration`)  
Change type: documentation-only consolidation; no worker, telemetry, controller, UI, scene, or admission change.

## Decision: infrastructure closed; actual MVS-runtime acceptance deferred

**I6.8A–F observational telemetry infrastructure and the bounded integration qualification are complete within the explicit evidence below.** The offline SfM producer has been exercised with a *real* bounded reconstruction worker. The production main-page Qt presentation has been exercised with *simulated* sidecars, and the production dense-worker wrapper, sampler, relay, controller signals, and main page have been exercised with a *test-controlled dense fixture*. These are distinct evidence sources; no test-generated dense stage is represented as actual MVS computation.

**Not qualified:** telemetry correctness, sensor coverage, UI responsiveness, or performance overhead while a genuine PyCOLMAP dense/MVS workload executes undistortion, PatchMatch, or fusion. The main application also has not been observed running a genuine offline reconstruction *concurrently inside the full GUI* as one end-to-end experiment. Accordingly this document **does not declare full real-MVS operational acceptance**. If project acceptance requires a genuine dense-run demonstration before considering all of I6.8 complete, that acceptance remains **OPEN**; do not relabel the fixture as a substitute. The infrastructure milestone can be checkpointed and I6.9 work can begin without claiming that deferred runtime evidence has passed.

## Evidence ledger (distinct scopes; counts are not additive)

| Gate | Workstation evidence and result | Precise scope and qualification boundary |
|---|---|---|
| A–E implementation | Commits `d4dbe05` (A), `303d111` (B), `324affe` (C), `c8dfc7e` (D), `5092282` (E); I6.8E: 106 targeted tests and 26 I6.7F focused tests passed; manual idle-page and details-dialog review completed. | Passive observation contract, real-stage adapters, resource sampler, bounded sidecars/Qt relay/session isolation and compact UI. No telemetry-controlled admission, persistence, overall completion percentage or ETA. |
| Bounded diagnostics | `scripts/run_i6_8f_telemetry_qualification.py --mode all`: 256 synthetic transport observations; poll median 0.2893 ms, p95 0.3517 ms, max 0.5154 ms; history 32; diagnostic CPU child/host RAM/disk/device-GPU sensors observed; probe exit 0. Checkpoint `61ddd3d`. | These are synchronous **synthetic poll** timings, not main-GUI FPS, matched telemetry-on/off cost, or SfM/MVS runtime overhead; diagnostic process readings are not attributed to reconstruction. |
| Full regression | `bash scripts/run_i6_7f_tests.sh closure`: 4,763 tests passed in 239.80 s; return code 0; focused and predecessor gates, compile, shell, and storage preflight passed. Run before checkpoint `61ddd3d`. | Automated suite on the operator's workstation. Later I6.8F deliveries are tests and documentation only; this result does not independently exercise a real dense runtime. |
| Genuine offline-worker lifecycle | 24 real TempleRing images copied to disposable scratch; CPU-only production SfM worker success; 11.702 s wall; actual resource sidecar observed while worker alive at 7.642 s; 15 Qt relay deliveries and **one real resource observation** including job-process RSS; stages extraction, matching, mapping, finalizing; no foreign-session delivery; terminal UI model cleared; probe exit 0. Checkpoint `f9508d8`. | Real offline SfM worker → actual sidecars → production Qt relay → presentation model. This **does not** prove simultaneous real job execution in `MainWindow` or any dense/MVS-stage observation. Initial eight-image attempts yielded zero resource samples before the bounded 24-image pass; they are not retroactively represented as passing. |
| Main-page Qt integration | `bash scripts/run_i6_8f_main_page_ui_tests.sh`: **11 passed**, exit 0; checkpoint `1f60729`. | Production `MainWindow`, Reconstruction page, controller, Qt relay/timer, details dialog, identity/staleness/terminal handling exercised with **simulated disposable telemetry sidecars**; no reconstruction starts. |
| Dense-path bounded integration | `bash scripts/run_i6_8f_dense_qualification_tests.sh`: **30 passed**, exit 0; checkpoint `6576a2f`. | Real dense-wrapper observation hooks, unchanged 5-s resource sampler, Qt relay/controller/page, run isolation and terminal cleanup tested with a **disposable, test-controlled MVS replacement**. Fixture intentionally exits non-success and cannot create/admit dense geometry. No actual undistortion, PatchMatch, fusion, or real dense GPU performance is qualified. |

All numbers above are transcribed from the operator-provided workstation terminal results; this closure document is an evidence ledger, **not an independent replay** of those runs. See the committed individual gate documents for their protocols and limitations:
`I6_8F_TELEMETRY_QUALIFICATION_GATE.md`,
`I6_8F_BOUNDED_REAL_WORKER_TELEMETRY_PROBE.md`,
`I6_8F_MAIN_PAGE_QT_QUALIFICATION.md`, and
`I6_8F_DENSE_WORKER_QT_QUALIFICATION_GATE.md`.

## Preserved operational constraints

- Telemetry is observational. Existing worker/controller terminal states and the independent dense verifier alone govern completion and admission; a sidecar, stage counter, provisional preview, or resource reading never admits geometry or establishes metric scale.
- Stage counters are local observed work, **not** overall progress, predicted completion, ETA, or fusion percentage. A device-wide NVIDIA GPU sample is **not** job-attributed GPU use. Unavailable/stale sensor values remain unavailable/stale rather than zero.
- Sidecars are latest-only; relay observations are session/dense-run isolated and bounded (32-entry history in the measured diagnostic). Terminal/new-run cleanup prevents obsolete readings from remaining live.
- No South Building fusion rerun, historical scene mutation, full TempleRing reconstruction, new persistence, cancellation/recovery behavior, hardware scheduling policy, or viewer/admission change was required for this qualification.

## Deferred acceptance: genuine MVS telemetry

**Status: OPEN — not a failed or passed real-MVS test.** At the first *independently justified and separately authorized* bounded dense job with legitimately admitted sparse input, collect direct evidence that observed undistortion/PatchMatch/fusion stages (as reached) and actual resource sidecars reach the live UI with correct session/run identity, scopes, freshness, terminal cleanup and no admission side effects. Keep the job's runtime/memory/disk safety controls and verify actual stage/worker state; if a stage is not reached, record it as unobserved. If claiming telemetry overhead or UI responsiveness under MVS load, use an appropriate matched baseline and explicit measurements rather than the synthetic poll microbenchmark. Do **not** schedule a job solely to turn this documentation gate green, do not synthesize an admitted sparse result, and do not replay South Building fusion. A genuine dense-runtime limitation must remain visible in future release/acceptance claims until observed evidence is recorded.

## Handoff and Git acceptance

This one-file closure ledger may be committed once its exact path, content and `git diff --cached --check` are reviewed against the clean Gitea baseline `6576a2f`. No automated suite or reconstruction is rerun for this documentation-only change. After a successful commit and push, confirm a clean branch tracking `origin/hotfix/v0.9.0-video-calibration-context`. Continue the locked roadmap with **I6.9 — cancel/pause/resume/recovery**, carrying the genuine-MVS telemetry acceptance item as an explicit deferred validation obligation. No merge into `main` is implied by this record.
