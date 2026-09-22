# I6.8F — bounded real-worker telemetry qualification (evidence pending)

Baseline: committed `61ddd3d` on the existing Gitea hotfix branch. This
narrow qualification addition **does not** modify a worker, controller, Qt page,
telemetry schema, admission, persistent scene or any historical session. Do not
mark overall I6.8 closed until actual operator evidence warrants it.

## Protocol and resource boundaries

`python scripts/run_i6_8f_real_worker_qualification.py` is **read-only by
default**. It checks one explicit existing real-image folder, selects 24
naturally ordered adjacent real images near its center, validates that they
are regular files, <=12 MiB each and <=1280 pixels per dimension, checks the
installed isolated PyCOLMAP **4.1.1** interpreter, and verifies available
host RAM/disk. If the historic TempleRing session is not available at its
known original `/tmp` location, pass `--images-dir` for a known existing
small, overlapping real-image folder; do not rehydrate/rebind relocated
South Building or TempleRing evidence to make this probe run.

With `--run`, the harness copies ONLY those selected images into its own fresh
`tempfile.TemporaryDirectory` and launches one **real production offline SfM**
worker on the copies. The profile is Efficient, CPU-only, <=960-pixel SfM
images, sequential matching with overlap 3, 24 keyframes max, and hard
**120-second wall-clock**, **5 GiB process-tree RSS**, **1 GiB disposable
scratch**, **4 GiB host MemAvailable** and **10 GiB available disk** guards.
The owner supervises and terminates only the process group it created.
The scratch directory and any provisional SfM output are deleted when the
probe ends, even if the production worker fails or is stopped by a guard.
This check cannot start dense, PatchMatch, fusion, training or persistent
scene creation. It does not modify or replay an original session.

During the real worker run, an actual `JobTelemetryQtRelay` consumes only the
production worker's real `job_progress_observation.txt` and
`job_resource_observation.txt` sidecars; `JobTelemetryPanelState` receives
real Qt signal deliveries, checks session identity, renders stage/resource
labels and clears terminal state. No synthetic job observations are injected.
A separately bound wrong-session receiver checks rejection of the genuine
sidecars. Output includes the actual stages seen, real process-sensor status,
exit status, stop reason and a short bounded log tail. A worker processing
failure **cannot** be called successful reconstruction/admission.

## Explicit limits of this check

This is a *real production worker → sidecar → Qt relay → presentation-model*
qualification, **not** a main-application ReconstructionController/MainWindow
interaction or GPU viewer/frame-rate performance measurement. A non-running
main-page screenshot, previous synthetic Qt tests, and this Qt relay result
must not individually be described as end-to-end real-worker UI evidence.
CPU-only SfM cannot qualify the explicit dense/MVS producer, nor can it
prove performance equivalence to an active GPU-backed workload. Do not
claim stage percentages, ETA, job-specific GPU VRAM, qualified scale or
fusion completion from observed telemetry.

If the default source is unavailable, an image is too large, runtime
version differs, sensors are unavailable or a resource guard triggers, the
harness stops and reports that condition. Do not work around the guard or
repeat expensive jobs just to obtain a PASS. Final closure evidence should
record which scope passed, which scope remains untested and the original
full regression (4,763 PASS at checkpoint `61ddd3d`) separately.

## I6.8F bounded timing-gap correction (pending workstation execution)

The initial **eight-image** production worker completed successfully and emitted
real stage observations with five Qt deliveries, but produced no resource
observation. Its short duration relative to the normal five-second delayed
sampler is a *timing hypothesis*, not a proven cause of that failure.
The observed result was `RESOURCE_OBSERVATIONS=0`,
`REAL_WORKER_TRANSPORT_PRESENTATION=INCOMPLETE`, probe return code 3.
Do not recast its independent safety-monitor RSS reading as a delivered
telemetry value. The first-run evidence must remain intact.

This correction changes **only this qualification harness, its tests and this
document**. Preflight now chooses 24 contiguous real images by default, still
within the same 12 MiB/image, 80 MiB total, 1280-pixel input bounds. Explicit
`--image-count N` is supported for 8..24, but do not increase beyond the
24-image default or loosen any resource/time limit to obtain a passing result.
A default preflight that cannot find 24 eligible images must stop safely.
Run this revised scenario **once** after successful preflight. It must not
start dense reconstruction or modify original evidence.

Additional fields report the worker wall time, whether a resource sidecar
existed while the harness worker was alive, the first sighting time, its final
presence/size, the number of *real* Qt resource deliveries, and whether a
job-process RSS reading was observed. The production resource sampler still
waits five seconds before its first sample; a delayed/failed sensor may take
longer. The probe never waits artificially after real SfM finishes and never
injects synthetic observations to obtain a PASS. Logs are shortened to keep
terminal output bounded. Even if this probe passes, main-application live UI
and dense/MVS producer remain **not tested** by this CLI diagnostic.
