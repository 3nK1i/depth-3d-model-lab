# I6.8D — Bounded telemetry transport, session isolation and history

Date: 2026-09-21. Baseline: `324affe` on `hotfix/v0.9.0-video-calibration-context` (Gitea). This step continues I6.8A's validated observational schema, I6.8B's factual stage sidecars and I6.8C's optional resource sidecar. It does not modify SfM, MVS, worker scientific outputs, independent admission, cancellation, pause/resume or scene persistence.

## Existing producers and the transport path

- The existing sparse PyCOLMAP adapter publishes its latest observational stage to `<offline_output_dir>/job_progress_observation.txt`; its optional sampler publishes the latest resources to `<offline_output_dir>/job_resource_observation.txt`.
- The existing explicit dense worker publishes its latest stage and resource observations to the corresponding two files under `<offline_output_dir>/dense_runs/<dense_run_id>/`.
- Both producers use the frozen `eia.reconstruction.job_telemetry.v1` event contract. A stage counter is a **number of observed work units**, not total percentage, ETA, or final/admitted geometry. GPU metrics are **device-wide**, not job-attributed.
- `JobTelemetrySidecarTransport` reads **only** those two job-local latest-only files and keeps a maximum of 32 parsed observations in RAM. Its Qt owner polls every 1500 ms, with at most two file reads and two observation signal emissions per poll; producers need not enqueue per-sample messages in the existing control protocol. New samples replace earlier samples, so intermediate samples may be skipped without violating per-producer sequence ordering.

## Admission, identity and invalid-input boundaries

The transport refuses symlinked or non-regular files, oversized/unreadable/malformed payloads, and samples older than 30 seconds or more than five seconds in the future. Each observation must match the **exact** bound job kind, source session UUID, optional dense run UUID and strictly increasing sequence **per producer**. The underlying contract also rejects extra fields that could masquerade as result, progress percentage or admission. Stale/latest readings become unavailable after 15 seconds even if a source file remains on disk; an unavailable reading is never relabelled as zero.

The frozen I6.7F `OfflineReconstructionController` is **not modified**: its owning application `ReconstructionController` binds an external relay when the established offline state changes to STARTING, after the controller has assigned the new session ID and output directory; it polls and closes that relay on terminal state. This also closes the relay when offline worker launch fails. The explicit dense controller binds the relay to the new run directory, closes it when the worker exits **before** the separate independent verification process starts, and closes it on any terminal state/shutdown. Old-run events cannot be forwarded into a new run after reset or completion. The application-level `ReconstructionController` forwards both sparse relay and dense-controller passive observations on `job_telemetry_received` without mixing them into sparse results, dense admissions, preview snapshots, scene state or job control. The main Reconstruction-page display is explicitly deferred to I6.8E.

## Validation evidence and remaining qualification

Tests-first red evidence: imports of the missing transport module and tests of the missing controller telemetry signals failed before implementation. The new tests cover missing and malformed files, symlink/size bounds, independent producers, foreign run and session, duplicate/out-of-order sequence, old and future timestamps, stale resource display, closure/reset and 32-event history bounds; a Qt relay test exercises delivery and reset without starting any reconstruction. Relevant I6.8A–C, dense-isolation and final-admission predecessor tests are included in `scripts/run_i6_8d_tests.sh`.

In the source inspection environment the Qt-free subset can run, but PyQt6 is unavailable, so **the Qt relay test is skipped there**. It must pass on the user's `depth` workstation, followed by the existing I6.7F focused regression. Continuous live hardware readings, long-running Qt event-loop overhead and professional UI display are not qualified here (I6.8E–F). No South Building/TempleRing reconstruction run is required.
