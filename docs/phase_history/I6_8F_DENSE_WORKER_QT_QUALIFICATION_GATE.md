# I6.8F — Dense worker telemetry qualification gate (bounded, no MVS)

Baseline: Gitea branch `hotfix/v0.9.0-video-calibration-context`, commit `1f60729`.
This delivery consists of **tests and test-only tools**; no production worker,
controller, sensor, reconstruction algorithm, viewer, or admission logic changes.

## What this check actually exercises

The standalone script `scripts/run_i6_8f_dense_telemetry_fixture.py` runs the
**production dense worker wrapper** in its own supervised test-owned process.
The expensive production `execute_explicit_colmap_dense_after_sparse` function
is explicitly replaced **in the disposable fixture process only**. Test-created
validated progressive snapshots drive the unmodified worker's observation hook
and latest-only progress sidecar. The worker's real `JobResourceSampler` samples
its fixture-owned process at the **unchanged five-second interval**. The real
`JobTelemetryQtRelay` 1500 ms timer carries these producer sidecars through
`ExplicitDenseJobController.telemetry_received`, the outer controller signal,
and the visible `MainWindow` Reconstruction-page telemetry window. The test
requires a sample **while the subprocess is alive**, followed by cleanup of
readings at a controller terminal signal. It checks session/run identity,
no fabricated fusion percentage, process-versus-device scopes, at-most-32
history, no admission file, and no change to authoritative sparse evidence.

The fixture deliberately raises `I6.8F_TEST_FIXTURE_NO_MVS_NO_ADMISSION` and
exits nonzero after bounded observation. Its run directory must be a fresh,
**empty, top-level** `/tmp/i68f_dense_fixture_*` directory, deleted by the
owner at test end. It does not read real imagery, load existing scene state,
run actual COLMAP, undistort an image, calculate a depth map, launch PatchMatch,
perform fusion, produce dense geometry, or verify/admit a final dense artifact.
The observed stages are **test-generated** and must never be described as real
MVS activity or performance. It does not validate production
`ExplicitDenseJobController.start()` against a genuine admitted sparse result;
test code explicitly binds the active controller identity rather than launching
an expensive production dense job. Real GPU-wide readings, if available, do
not constitute job-attributed VRAM usage.

## Existing separate evidence

- At `f9508d8`, one genuine eight-then-24-image **offline SfM**, CPU-only
  worker probe qualified the SfM stage/resource sidecar -> Qt relay ->
  presentation-model lifecycle; successful 24-image run lasted 11.702s,
  produced one real resource observation, 15 Qt deliveries, no wrong-session
  delivery, and cleared at terminal. This was **not** an MVS run.
- At `1f60729`, 11 MainWindow/Qt integration tests with **simulated**
  sidecars qualified main-page presentation, scopes, run/session isolation,
  staleness, and terminal cleanup. They did not measure simultaneous real
  reconstruction inside the application.
- Earlier bounded transport microbenchmark and full 4,763-test regression
  passed separately. Neither demonstrates MVS telemetry during a real fusion.

## Closure decision

Only after workstation tests run and pass, record the results and checkpoint
this narrow qualification. An **I6.8 infrastructure implementation** may be
considered integrated after all A–F contracts and tests pass; **real dense/MVS
runtime telemetry remains unqualified** until observed in the first naturally
scheduled, separately authorized bounded dense job with legitimate admitted
sparse inputs. Do not rerun South Building, synthesize an admitted sparse
result, or convert these test-generated stages into production evidence merely
to get a green milestone. If strict real-MVS acceptance is required for I6.8
closure, leave that acceptance **pending** instead of declaring full closure.

Run `bash scripts/run_i6_8f_dense_qualification_tests.sh`; require real Qt
execution (not skipped), test exit zero, and diff integrity. Do not repeat full
reconstruction/regression when this tests-only addition succeeds. No Git
commit/push until workstation results are reviewed.
