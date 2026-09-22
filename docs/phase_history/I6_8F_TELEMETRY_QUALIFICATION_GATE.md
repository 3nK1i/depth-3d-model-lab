# I6.8F — Telemetry performance and real-runtime qualification gate

Date: 2026-09-21. Prerequisite: clean Gitea checkpoint `5092282` on
`hotfix/v0.9.0-video-calibration-context`. This qualification-only change
adds a bounded diagnostic harness, tests, and evidence protocol; it does not
change reconstruction, scene persistence, admission, the viewport, or GUI logic.

## Qualification is evidence-gated, not a declaration of completion

I6.8A–E established a strict observational schema, real stage adapters,
optional passive resource sampling, bounded sidecar transport, session/run
filtering, and a compact main-page telemetry UI. The dedicated TempleRing
viewer and 163,827 admitted points remain previously qualified I6.7 evidence,
**not** real I6.8 job telemetry evidence. The failed South Building fusion is
not restarted. No telemetry may promote a provisional preview, imply certified
scale, infer an overall percentage/ETA, or determine SfM/MVS success.

### Gate 1 — Workstation automated regressions

Run `bash scripts/run_i6_8f_tests.sh` in `depth`, then
`bash scripts/run_i6_7f_tests.sh focused`, then the existing full-unit
regression via `bash scripts/run_i6_7f_tests.sh closure` when preparing final
I6.8 closure. Record return codes and actual counts. The source-inspection
environment does not have PyQt6 and cannot stand in for workstation GUI tests.

### Gate 2 — bounded diagnostic process + host sensor measurements

Run `python scripts/run_i6_8f_telemetry_qualification.py --mode all`.
The **transport mode** uses 256 explicitly synthetic diagnostic observations,
a single temporary directory, two sidecars of <=4097 bytes each, a 32-entry
history cap, and measures local synchronous `poll()` latency with Python
`perf_counter_ns`. Its p95 and maximum are **not** Qt rendering FPS or
reconstruction overhead, and tracemalloc peak is not host RSS.

The **host mode** launches only one probe-owned, short-lived Python CPU child
with an 8 MiB buffer, samples its process tree three times, reads *real* host
MemAvailable and workspace disk free, and attempts a bounded NVIDIA-device
query. Unavailable sensor readings remain unavailable, not zero. The one-child
CPU/RSS readings are attributed only to the diagnostic process, never to an
SfM/MVS worker. The stage value used to exercise the sampler in this mode is
**synthetic** and does not qualify live source ingestion or any actual
reconstruction stage. The diagnostic child is terminated only by its own
owning harness and is reaped even on probe failure. No dataset, original
session, GPU training, SfM, PatchMatch or fusion is launched.

Collect the actual measured p95, maximum, tracemalloc peak, process-CPU/RSS
availability, host RAM/disk availability, and device-wide GPU sensor status.
Do not invent numerical performance pass thresholds after the fact; unexpected
latency/overhead is grounds for investigation, not an automatic PASS.

### Gate 3 — representative **actual** reconstruction worker lifecycle

**Still required before declaring full I6.8 closure.** The diagnostic harness
only establishes transport and sensor behavior; it does not supply observed
telemetry from a real active reconstruction job. If a short, bounded existing
fixture can be used safely on the current workstation, separately verify:

- The real SfM worker emits observed stages into its run-specific sidecar;
  telemetry is delivered to the main UI for the correct session and stops on
  terminal state, including restart/new-session isolation.
- The explicit dense worker, only if a bounded provider invocation is
  affordable, reports its actual stage and process-tree usage without
  misreporting device-wide GPU utilization/VRAM as job usage. Neither the
  original nor a recovery of South Building is required.
- GPU readings and CPU/RAM/disk readings have actual scope/unit/availability,
  stale samples disappear, malformed or foreign-run samples never appear, and
  no observed counter is presented as an overall fraction or ETA.
- Measure representative sampling/UI responsiveness and compare against a
  **matched** telemetry-disabled or equivalent baseline if claiming telemetry
  overhead; do not call one diagnostic `poll()` measurement a throughput or
  frame-rate result.

If a representative job cannot be run without expensive reconstruction, mark
**real-worker integration and overall I6.8 closure pending**, preserve the
clean existing I6.8A–E checkpoint, and proceed only with explicitly scoped
qualification evidence. Do not relabel synthetic inputs as live observations.

### Gate 4 — final documentation and Git

Record exact workstation output and manual UI findings, unresolved limitations,
full regression result, and verify the current Git file set and diff. Commit to
the existing Gitea branch only after the relevant gate passes. I6.9
cancel/pause/resume/recovery and I6.10 persistence/reopen/resume remain separate
locked roadmap steps, not implied by this diagnostic probe.
