# I6.8F — Main-window Qt telemetry qualification (scope: page integration)

Prerequisite: Gitea checkpoint `f9508d8` on `hotfix/v0.9.0-video-calibration-context`.

This patch adds **tests only**: the production controller, telemetry schema,
worker, resource sampler, main GUI and admission logic are unchanged. The tests
construct the **actual MainWindow**, switch to the real Reconstruction page,
and exercise its real controller/relay Qt signals and scheduled relay poll timer.
The test sidecars are expressly **simulated**, stored in pytest's disposable
temp directory: they do not come from a running reconstruction and their CPU/GPU
values are not workstation measurements. No image dataset or SfM/MVS job starts.

## Acceptance boundaries

- The real MainWindow's offline page binds the current session; the real Qt
  relay timer reads simulated stage and resource sidecars and forwards them
  through the *production* controller signal into the visible telemetry dialog.
- The page distinguishes job-tree CPU/RAM, host RAM/disk and **device-wide** GPU
  readings, with explicitly unavailable and stage-only counter wording.
- Terminal-state signal clears the page; old or foreign sessions cannot refill
  it. Freshness rejection prevents old values being presented as live.
- Dense job state and the production dense relay deliver a run-matched simulated
  stage, reject late old-run samples, and clear at terminal state. This test
  **does not qualify actual MVS production observations or execution**.
- Existing runtime controls remain children of the existing runtime card.

These tests complement, **do not replace**, the previously qualified genuine
24-image offline worker -> resource sidecar -> Qt relay -> presentation-model
run at checkpoint `f9508d8`. Together they cover the major offline links, but
**simultaneous real-worker-in-full-GUI operation remains unmeasured**. The UI
integration test is not a Qt frame-rate or job-overhead benchmark. No fabricated
telemetry is injected into normal user application sessions.

## Workstation validation

Run `bash scripts/run_i6_8f_main_page_ui_tests.sh` in the `depth` environment
with a usable Qt display; ensure Qt and pytest-qt tests are **executed, not
skipped**. Check `I6_8F_MAIN_PAGE_UI_TEST_RC=0` and inspect the test count; run
`git diff --check` and inspect the exact three-file new set. If UI tests fail,
fix the narrow cause; do **not** rerun offline reconstruction. Preserve the
previous 4,763-test full regression result; only a production-code change
would warrant an expanded regression gate.

## Remaining before final I6.8 closure

Bounded actual dense-worker telemetry qualification (no South Building fusion
or large reconstruction), clear disclosure of any untested full-GUI simultaneous
real-worker path, and a final evidence-gated closure commit. Never infer dense
qualification from these simulated sidecars.
