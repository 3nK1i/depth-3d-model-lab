# I6.8C — Bounded, observational CPU/RAM/GPU/storage telemetry

Date: 2026-09-21. Source baseline: Gitea development branch
`hotfix/v0.9.0-video-calibration-context`, I6.8B commit `303d111`.

## Contract and actual runtime ownership

- I6.8C adds the optional `JobResourceSampler` to the **existing** sparse
  `ExternalPycolmapSfmBackend` (bound to the isolated PyCOLMAP worker PID), and
  the **existing** explicit post-sparse dense Python worker (bound to its own
  PID, including its MVS subprocess descendants). This does not enable Auto,
  change the reconstruction workflow, or start any additional SfM/MVS job.
- The sampler runs in one daemon thread **per live job** with a minimum 2 s,
  default 5 s sample interval. It waits one interval before its first poll,
  emits a *latest-only* `job_resource_observation.txt` sidecar into the live
  session/dense-run directory, and stops when the owning call returns or fails.
  There is no unbounded queue or continuously growing log. The sensor probe
  uses a subprocess timeout of 1.5 s for each bounded `nvidia-smi` query.
- An observation is job-scoped through the existing I6.8A session UUID and
  optional dense run UUID. Its independent producer name is `resource_sampler`
  and its sequence advances on observed samples only. A stage is assigned
  **only after the existing worker stage event is seen** (or, for dense, after
  the actual `dense_starting` lifecycle event). There is no overall progress
  fraction, ETA, final-admission claim or process-control directive.
- I6.8B's `job_progress_observation.txt` remains separate and unchanged.
  Neither diagnostic file is a persistent scene or an authoritative result.
  I6.8D implements the bounded UI transport, staleness and session filtering;
  I6.8E presents it in the GUI.

## Metric definitions and limitations

| Metric | Scope | Observation and units |
|---|---|---|
| `process_tree_rss_bytes` | Job PID and descendants | Sum of each currently resident process's RSS; **not unique physical RAM**. Missing psutil, disappearing processes or a tree beyond 128 descendants yields `unavailable`, never invented zero. A reused root PID is rejected using process creation time. |
| `process_tree_cpu_percent` | Job PID and descendants | Delta of cumulative user + system CPU seconds divided by elapsed monotonic seconds, expressed as percent of **one logical CPU**. May exceed 100 on parallel tasks. First sample or an invalid delta is `unavailable`. |
| `system_ram_available_bytes` | Entire host | Linux `/proc/meminfo` `MemAvailable` only, or `unavailable`. Not attributed to this job or substituted with `MemFree`. |
| `workspace_disk_free_bytes` | Filesystem hosting the run directory | Available bytes from `shutil.disk_usage()`. Shared with other processes; not job-specific allocation. |
| `gpu_device_util_percent`, `gpu_device_vram_used_bytes` | Entire NVIDIA GPU device | `nvidia-smi` utilization and used VRAM, at most four enumerated devices. **Not** job GPU usage or job GPU allocation. Missing/bad sensor readings are null/unavailable; an unavailable probe with index 0 does not assert device 0 exists. |

The v1 I6.8A schema admits **at most 12 readings per event**. Four process/
host fields plus two GPU fields for at most four devices obey that limit.
Sampling can observe resource pressure but is **not** an OOM guard, scheduling
policy or an exact peak measurement. The original MVS worker's stage-boundary
resource snapshots remain separate and unchanged. No process-specific VRAM
is emitted: `gpu_process_vram_used_bytes` remains unavailable until a
correctly attributed source is separately qualified.

## Fail-soft, verification and excluded work

Resource sampling is optional. Missing psutil, NVIDIA tools, permissions,
readings, or a writable run directory do not invalidate real SfM/MVS output.
The sampler does not change the original dense snapshot protocol, worker
termination, cancellation, progress fractions or independent final admission.
The mistakenly removed executable bit of
`scripts/run_i6_7e2b2a_post_sparse_dense_worker.py` from I6.8B is restored in
this delivery; its original CLI and behavior are otherwise preserved.

The I6.8C tests cover observed/unavailable scope, time-delta CPU, summed RSS,
missing GPUs, bounded GPU output, root PID reuse, source-hook isolation,
fail-soft sidecar, and invalid stages, plus I6.8A/B, existing SfM and MVS
predecessor regressions. They do not qualify CUDA utilization accuracy,
main-page UI latency/FPS, aggregate runtime overhead, or real multi-GPU
hardware. Those require later I6.8D–F qualification.

No South Building fusion, TempleRing reconstruction, session relocation
(I6.10), pause/resume (I6.9), hardware scheduling (I6.11), metric-scale
promotion, main-branch merge, or persistent scene work is in scope.
