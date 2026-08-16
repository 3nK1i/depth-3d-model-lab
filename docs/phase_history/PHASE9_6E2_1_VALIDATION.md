# Phase 9.6E2.1 — Recorded Video Offline Progress/FSM Correction

## Scope

Narrow compatibility correction for the existing Offline Reconstruction `Start` workflow exposed by longer recorded-video input. Phase 9.6E2 Full-Scene DA3 is unchanged.

## Root cause

The protected `ReconstructionEventRecorder` correctly requires progress to remain monotonic within one reconstruction stage. The isolated PyCOLMAP worker uses `incremental_mapping(..., multiple_models=True)`. PyCOLMAP may therefore invoke `initial_image_pair_callback` more than once when it seeds disconnected candidate sparse models.

The worker previously emitted every initial-pair callback as mapping progress `0.68`. After an earlier model had already registered images and advanced progress above `0.68`, a later model seed reported `0.68` again and the application correctly rejected that rollback with:

`Progress must be monotonic within one stage unless restart is explicit.`

## Correction

- Keep the global monotonic-progress validator unchanged.
- Track mapping progress inside the isolated PyCOLMAP producer.
- The first initial-pair callback advances mapping to `0.68`.
- Additional model-seed callbacks are status-only and carry no progress fraction.
- Registered-image callbacks are clamped to the last emitted mapping progress and remain bounded by `0.92`.
- Worker metadata records initial-pair and registered-image callback counts for diagnosis.

## Authority and runtime boundaries

No reconstruction authority, scale, measurement, QC, persistence, DA3, Gaussian, NeRF, or resource-coordination policy changes are made. PyCOLMAP remains isolated in its external runtime.

## Qualification

Automated gates cover repeated multi-model callback ordering, the external worker protocol boundary, recorded-video SfM materialization, worker lifecycle, the original event invariant, offline controller compatibility, and Phase 9.6E2 DA3 compatibility.

Manual qualification should use the same recorded video that previously triggered the rollback, select `Recorded video`, then use the normal upper Offline Reconstruction `Start` action. The run must advance beyond sparse mapping without the monotonic-progress error. Existing arbitrary-scale/metrology status must remain unchanged.
