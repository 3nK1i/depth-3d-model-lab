# I6.2D — Professional One-Start UX & I6.2 Closure

## Purpose

I6.2D closes the One-Start Reconstruction product slice without expanding the
qualified reconstruction boundary. It improves only the user-facing semantics
around the already-qualified I6.2A/B/C path.

## Product behavior

- A ready image-sequence or recorded-video source exposes one **Start** action.
- While an offline run is active, duplicate Start remains blocked by the
  existing controller/page enablement logic.
- Offline pause/resume is explicitly described as unavailable at this stage;
  **Cancel** remains the qualified stop action.
- A completed, admitted sparse SfM result is presented as **Sparse reconstruction
  complete** and **Partial result**, not as a final dense/canonical scene.
- A completed sparse result exposes a **Run again** affordance that starts a new
  isolated session/output identity; it does not overwrite the completed session.
- If the worker completes but I6.2C rejects the product handoff, the rejection
  remains visible as an error instead of being overwritten by the worker's
  terminal `COMPLETED` state.
- Preflight errors remain user-facing after control refresh.

## Authority boundaries preserved

I6.2D does not modify the ReconstructionController, offline worker, persistence
format, providers, reconstruction algorithms, scientific thresholds, or output
artifacts. PyCOLMAP remains canonical camera/global-frame authority. Sparse SfM
remains a partial result. Dense execution and persistent-scene promotion are
not authorized. Production Auto remains unqualified/unauthorized and certified
metrology is not claimed.

Progressive sparse visualization is deliberately not added here; it remains an
I6.6 responsibility.

## Validation

The closure runner covers:

- dependency-light I6.2D presentation tests;
- real Qt page/control semantics;
- I6.2C handoff regression;
- I6.2B orchestration regression;
- I6.2A preflight regression;
- I6.1 workspace regression;
- Phase 8 runtime/page predecessor tests;
- complete unit regression in `closure` mode.

A final manual UI run is required before declaring I6.2 closed.
