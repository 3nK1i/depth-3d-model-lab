# I6.7F — One-off South Building persisted-sparse recovery

This utility is intentionally narrow. It **does not rehydrate the GUI session** and does not start SfM, mutate the admitted sparse result, relabel a failed run, relax final checks, or enable Auto. It uses the existing Qt-owned `ExplicitDenseJobController` in a `QCoreApplication` event loop; that controller creates a fresh UUID, launches the isolated COLMAP-MVS worker in its own process group, and invokes the independent terminal artifact verifier. Keep the launch terminal open; Ctrl+C requests supervised cancellation.

Arguments: `--offline-output-dir` (persisted sparse session directory), `--session-id` (same accepted session UUID), `--confirm-explicit-colmap-mvs` (required). Default provider timeout 360 minutes; 720-minute hard ceiling applies.

This recovery script limits its use to the known accepted 87-view South Building source. It requires at least 35 GiB free at start, automatically requests supervised cancellation below 10 GiB, and writes a per-new-run log in `dense_recovery_logs/`. That log does not serve as final admission evidence. The supervisor emits `INDEPENDENT_FINAL_ADMISSION=PASS` only on successful independent verifier completion. Later open the resulting artifact in the existing full-viewer demonstration with the fresh run UUID, manifest hash, and geometry hash printed by the log/admission.

Because the previous GUI was closed, the in-app dense progress bar will not reflect this one-off recovered job. Normal in-app progress remains available for jobs started from an active sparse session in the GUI. This utility should not be described as general persisted-session restoration or as automatic retry.
