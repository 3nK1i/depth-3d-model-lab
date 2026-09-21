# I6.6C — Controller Transport, Session Isolation & Backpressure

## Purpose

I6.6C transports the provisional sparse-SfM snapshots produced and qualified in I6.6B from the isolated PyCOLMAP process into application-owned controller state. It does not render the snapshots and does not promote them to final reconstruction authority.

The frozen I6.6A snapshot contract remains the semantic authority. The frozen I6.6B worker remains the producer.

## Runtime path

```text
PyCOLMAP worker
    |
    | EIA_SFM_PROGRESSIVE <validated I6.6A payload>
    v
ExternalPycolmapSfmBackend
    |  session/request/revision validation
    v
run_reconstruction_offline_job.py
    |  dedicated progressive line; not ordinary offline status/progress
    v
OfflineReconstructionController
    |  latest-only pending buffer + 50 ms coalescing
    v
ReconstructionController
    |  passive provisional state
    v
(no viewport/UI consumer in I6.6C)
```

## Transport and authority rules

- `EIA_SFM_PROGRESSIVE` remains separate from the existing `EIA_SFM_EVENT` protocol and from the outer `eia.reconstruction.offline.v1` event stream.
- Every progressive payload is parsed through `ProgressiveSfmSnapshot.from_payload`; malformed or authority-upgraded payloads are rejected fail-soft.
- The inner backend requires the active session ID and exact request fingerprint before forwarding a snapshot.
- The I6.6A transition validator enforces strictly increasing revisions. Revision gaps remain valid so transport coalescing can discard superseded intermediate updates.
- The outer process bridge maintains at most one pending snapshot. A newer valid pending update replaces the older pending update rather than growing an unbounded queue.
- The coalescing interval is 50 ms and is strictly a UI/process-transport protection. It does not alter SfM, mapper thresholds, camera registration, or scientific output.
- `FINALIZING` bypasses the delay and is flushed immediately.
- A new run resets progressive transport state to its new session ID. Cross-session, cross-request, stale, duplicate, and malformed snapshots never replace current state.
- Terminal admitted offline reconstruction output supersedes and clears provisional controller state.
- Progressive transport failure does not fail an otherwise valid final SfM run.

## Deliberately excluded from I6.6C

- No incremental VisPy rendering.
- No reconstruction-page UI additions.
- No camera/landmark viewer mutation.
- No dense output or progressive dense visualization.
- No provider-selection or production-Auto changes.
- No frame/view replanning during SfM.
- No persistent canonical scene creation.
- No metric-scale or metrology claim.

Incremental sparse viewer integration is reserved for I6.6D. Professional UI/performance qualification and I6.6 closure remain reserved for I6.6E.

## Qualification

I6.6C requires:

1. focused pure/static/backend/backpressure tests;
2. focused Qt process/controller transport tests;
3. I6.6B/I6.6A plus SfM/I6.5/I6.4 regression boundaries;
4. complete unit-suite regression;
5. Python compile and shell syntax checks;
6. a real PyCOLMAP/QCoreApplication transport smoke proving provisional snapshots cross both process boundaries, preserve session/revision authority, reach the application controller path, and leave final-result admission unchanged.

No manual UI qualification is required because I6.6C intentionally does not modify the viewport or reconstruction-page UI.
