# I6.6E — Professional Progressive SfM UX, Performance Qualification & I6.6 Closure

## Scope

I6.6E is the final productization/qualification layer over the frozen I6.6A-D progressive sparse-SfM pipeline. It adds no new reconstruction path and does not alter PyCOLMAP execution, worker emission, transport admission, final-result admission, dense reconstruction, provider selection, or production Auto.

## Product presentation

A compact `PROGRESSIVE SFM` disclosure group is added immediately after `FRAME & VIEW PLAN` in the promoted Reconstruction settings stack. It reports:

- lifecycle status (`Waiting`, `Live provisional`, `Finalizing provisional`, terminal state);
- mapping/finalizing stage;
- total registered cameras versus bounded camera preview samples;
- total sparse landmarks versus bounded landmark preview samples;
- current reprojection-error observation when supplied by PyCOLMAP;
- snapshot revision and provisional progress;
- I6.6C delivered/coalesced/rejected transport counters;
- UI-thread viewer-update cost (not GPU render latency);
- bounded NumPy preview payload size;
- explicit PyCOLMAP/arbitrary-scale authority and non-final/non-persistent/non-metric boundary.

The last provisional statistics may remain visible after a terminal state only as `Last provisional` telemetry. They are never relabeled as the admitted final sparse result.

## Performance semantics

I6.6E observes presentation cost without creating a reconstruction gate:

- Viewer update target: 50 ms. This aligns with the existing I6.6C coalescing interval and is a UI responsiveness target only. The recurring bounded projection/apply path is measured; the one-time initial viewport fit is excluded.
- Preview payload budget: 128 KiB for the NumPy camera/landmark/RGB buffers under the frozen I6.6B limits (64 cameras, 4096 landmarks).
- The performance accumulator is constant-space and retains no per-revision timing history.
- Exceeding the UI target is presentation evidence only and cannot reject, retry, replan, or alter SfM.

## Authority boundary

Progressive snapshots remain:

- provisional visualization only;
- PyCOLMAP camera/global-frame evidence;
- arbitrary scale;
- not final-result admission;
- not a persistent canonical scene;
- not metric scale;
- not measurement authority;
- not certified metrology.

The existing I6.2 terminal-result admission path remains authoritative for the final sparse reconstruction.

## I6.6 closure boundary

I6.6 closes only progressive **sparse** SfM visualization and UX. Progressive dense reconstruction remains I6.7. Surface mesh is I7 and Gaussian Splatting is I8 under the revised roadmap.
