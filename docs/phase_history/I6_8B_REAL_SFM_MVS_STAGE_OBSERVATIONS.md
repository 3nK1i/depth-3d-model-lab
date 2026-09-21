# I6.8B — Observed SfM/MVS stages, without invented job completion

Date: 2026-09-21. Base: committed I6.8A (`d4dbe05`) on `hotfix/v0.9.0-video-calibration-context`. Scope: a bounded, observational stage adapter and two additive producer hooks. This is **not** the I6.8D Qt transport or I6.8E UI.

## Source evidence and stage policy

- `core/reconstruction/sfm_pycolmap.py` receives *real, already-parsed* `SfmWorkerEvent` objects from the existing isolated PyCOLMAP process. The new passive observer maps the reported feature-extraction, matching, mapping, conversion and worker-complete messages to `sfm_*` stages. Its new sidecar is enabled only when the expected One Start layout has `sfm_workspace` immediately under the session-ID directory. The existing worker stdout, SfM event protocol, camera-pose authority, sparse result, and terminal lifecycle are unchanged.
- `registered_callbacks` and `initial_pair_callbacks` count worker callbacks, **not** verified distinct registered cameras. They never produce `registered_views` progress. Only a successfully converted typed sparse result provides `registered_keyframes`; the observer compares that count with the selected source-view population and labels it as a backend-reported count, **not** offline admission.
- `scripts/run_i6_7e2b2a_post_sparse_dense_worker.py` wraps the existing `ProgressiveDenseTransportLineSink` without replacing its original `publish` or altering `app/reconstruction_dense_production_worker.py`, `core/reconstruction/mvs_provider.py`, its runner, or the frozen preview/admission path. A *validated* progressive dense snapshot carries the actual `stage_detail` from the provider's polling callback. The observer maps only the known launch, undistortion, PatchMatch, stereo-fusion and worker-complete stage details.
- For PatchMatch, `geometric_depth_maps_observed` is the number of `.geometric.bin` files the existing provider's polling callback found in this clean run workspace. It is **not** a percentage of PatchMatch time or total reconstruction. `processed_views` on a fusion-stage preview still reflects that earlier map count, so **fusion has no counter, completion percentage, or ETA**. An isolated MVS worker's reported completion means that independent dense verification/admission is *pending*, not that geometry is admitted.
- The new `JobTelemetryObservation` contains no progress fraction, duration estimate, final-admission authority, scale qualification, resource measurement or process-control instruction. Legacy workflow fractions remain in their original event path for backward compatibility and are not imported into the observational contract.

## Producer and output boundary

The two hooks send observations only to a small, latest-only *diagnostic* file (one bounded telemetry line), rather than injecting another line into the existing Qt control protocol:

- Sparse: `<offline_session_output>/job_progress_observation.txt`.
- Dense: `<offline_session_output>/dense_runs/<dense_run_id>/job_progress_observation.txt`.

The file is replaced atomically only if the parent job directory already exists, and it retains only the latest observation. A sensor, mapping or filesystem failure is **fail-soft** and must not alter the original SfM/MVS run or its final result. These diagnostic sidecars are *not* persistent reconstruction scenes, admission manifests, durable telemetry history, or a resume mechanism; they must not be consumed as scientific authority. They do not make the main UI show new telemetry yet. I6.8D will implement bounded Qt transport and stale/session gating, and I6.8E will supply the visible job panel.

## Qualification and limitations

- Tests-first red state: importing the absent I6.8B observer failed during collection, before implementation.
- Local Qt-free checks: the B stage tests plus A contract and representative MVS/dense-admission predecessor tests passed; Python compilation passed. Source-hook tests exercise fail-soft sparse output and the dense worker's original transport with and without an observable sidecar. They do **not** start a new real reconstruction, establish a sustained telemetry sampling rate, or measure UI FPS/overhead.
- The isolated inspection environment has no PyQt6; thus I6.7F's UI test collection cannot run there. Its frozen-source/scope/compile/shell gates passed. Re-run `bash scripts/run_i6_7f_tests.sh focused` in the user's qualified `depth` environment before accepting this patch.
- Older I6.7C and I6.7E2B2A frozen-hash gates refer to earlier versions of `app/reconstruction_controller.py` already superseded in the committed I6.7F checkpoint. Their historical hash mismatch is not evidence of I6.8B source drift. Do not rewrite those historical hashes or interpret those scripts as the current I6.7F acceptance gate.

## Exclusions

No new SfM/MVS execution, South Building fusion, simulated advancement, resource polling (I6.8C), GUI transport and throttling (I6.8D), UI panel (I6.8E), cancel/pause/resume (I6.9), session relocation (I6.10), GPU scheduling (I6.11), metric-scale promotion, preview-to-final promotion, or main-branch merge. The preserved TempleRing/South Building sessions are untouched by this patch.
