# I6.8E — Professional observational job-telemetry UI

Baseline: `c8dfc7e`, branch `hotfix/v0.9.0-video-calibration-context`.
Scope: one compact passive telemetry group inside the existing Reconstruction
runtime card. The group subscribes to I6.8D's existing `job_telemetry_received`
signal and consumes only validated I6.8A `JobTelemetryObservation` objects.

## Visible evidence

- Active sparse and explicit dense jobs are identified separately; each new job
  resets the presentation cache and rejects observations from a previous job.
- Shows the observed worker stage and exact stage-local counter when available.
  `fusion_images_started` means *started*, not completed. Never compute overall
  job percent or ETA from counters. Existing sparse/dense progress bars are unchanged.
- Worker-process-tree (or explicitly worker-process) CPU/RAM readings are distinct
  from host-available RAM/workspace free disk and device-wide GPU utilization/VRAM.
  Unavailable values are not silently reported as zero.
- The existing bounded runtime timer re-renders the passive presentation, so
  samples older than the I6.8D freshness limit turn stale even without new signals.
- Terminal and new-run transitions clear the last live readings. Completion and
  admission remain entirely owned by the original job controllers/verifier.

## Qualification boundaries

This change does not start any reconstruction, write telemetry history, create
persistent scenes, produce metrical measurements, modify GPU viewer resources,
change auto selection, or qualify sustained FPS. The underlying sidecars remain
latest-only and worker-owned. A real active job must be observed in I6.8F; Qt
headless integration must pass in the workstation `depth` environment (local
source-inspection environment lacks PyQt6).

## Manual acceptance layout correction (2026-09-21)

The idle desktop screenshot at 1649×928 exposed a height regression: adding
six inline telemetry rows to the runtime card squeezed Start/Pause/Stop and
the explicit dense/full-view controls. The telemetry detail labels now live
in a modeless, read-only details dialog opened by a compact **Telemetry**
button alongside the existing dense-stage label. The main card keeps its
original controls and the real stage status remains available via the button
text/tooltip; live observations continue to refresh the dialog in place.

This is presentation-only: it does not alter job lifecycle, admission,
preview budgets, event sequencing, measurements, or telemetry sampling.
The correction adds static layout guards and an on-demand Qt dialog test.
Final visual acceptance at the operator's display size remains required
before committing I6.8E.
