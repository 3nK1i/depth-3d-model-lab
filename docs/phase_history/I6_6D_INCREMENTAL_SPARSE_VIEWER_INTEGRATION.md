# I6.6D — Incremental Sparse Viewer Integration

## Scope

I6.6D connects the already-qualified I6.6C provisional sparse-SfM controller stream to the existing Reconstruction VisPy viewport. It does not change PyCOLMAP execution, progressive worker emission, transport/backpressure, final-result admission, dense reconstruction, provider selection, production Auto, calibration, scale authority, measurement authority, or persistent canonical-scene authority.

## Runtime boundary

The production flow is now:

```text
PyCOLMAP worker
  -> I6.6B bounded provisional snapshots
  -> I6.6C validated/session-isolated/coalesced transport
  -> ReconstructionController.progressive_sparse_snapshot_changed
  -> I6.6D presentation binding
  -> bounded NumPy viewer frame
  -> ReconstructionCanvas replace/update buffers
```

Every progressive frame remains `provisional_visualization_only`. The admitted terminal sparse result continues through the pre-existing offline result handoff and `show_offline_sparse_reconstruction` path.

## Viewer projection

`app/reconstruction_progressive_sfm_viewer.py` is dependency-light and Qt-free. It converts one validated `ProgressiveSfmSnapshot` into read-only contiguous NumPy arrays containing only the bounded camera/landmark samples already authorized by I6.6A/B. No PyCOLMAP object crosses into the Qt process.

The projection preserves:

- session ID and strictly increasing revision;
- mapping/finalizing stage;
- sampled `T_reconstruction_from_camera` matrices;
- sampled sparse landmark positions and RGB where uniformly available;
- full registered-camera/landmark/observation totals as telemetry metadata;
- arbitrary reconstruction gauge and provisional authority.

It does not reconstruct omitted geometry, infer physical scale, or convert the preview into canonical geometry.

## Replace-only rendering

`ReconstructionCanvas.show_progressive_sparse_reconstruction` replaces the bounded sparse-landmark and registered-camera buffers for each admitted revision. It does not append old snapshot geometry. This prevents historical snapshot accumulation and keeps GPU/CPU presentation memory bounded by the existing display budgets.

The existing sparse landmark and keyframe-frustum primitives are reused. I6.6D adds a dedicated provisional camera-path line derived from the current sampled registered-camera positions. That line is distinct from the admitted offline camera path and is cleared at the provisional boundary.

The viewport is fitted once when the first non-empty provisional geometry appears. Later revisions preserve the user's orbit/pan/zoom state rather than repeatedly resetting the camera.

## Final-result handoff

A terminal offline result still supersedes provisional state through the existing I6.2/I6.6C admission lifecycle. `show_offline_sparse_reconstruction` starts from `clear_live_state`, which clears I6.6D provisional metadata/path before loading the admitted final sparse visualization.

A late/null progressive-clear notification is guarded so it cannot erase an already-loaded admitted offline sparse result.

## User-visible state

I6.6D adds only the minimum authority-safe viewport state text:

```text
Progressive sparse SfM • Mapping • provisional
```

or:

```text
Progressive sparse SfM • Finalizing • provisional
```

Professional status composition, richer progress presentation, performance qualification, and final I6.6 UX closure remain I6.6E.

## Preserved boundaries

- PyCOLMAP remains sole camera/global-frame authority.
- Progressive geometry remains provisional and arbitrary-scale.
- No progressive snapshot is admitted as the terminal sparse result.
- No persistent canonical scene is created.
- No metric or certified-metrology claim is made.
- I6.6A contract is unchanged.
- I6.6B worker/emission implementation is unchanged.
- I6.6C transport/controller/backpressure implementation is unchanged.
- No progressive dense visualization is added; that remains I6.7.
- Provider selection and production Auto are unchanged.
- I6.5 frame/view planning is not rerun during SfM.

## Validation plan

Automated closure covers:

- dependency-light viewer-frame projection;
- read-only/contiguous bounded buffers;
- replace-only/no-history static guards;
- Qt page/canvas progressive binding;
- first-fit-only behavior;
- mapping -> finalizing provisional transition;
- provisional clear;
- admitted offline sparse replacement;
- I6.6A-C regressions;
- existing sparse/offline viewport regressions;
- existing page/runtime regressions;
- full unit suite.

A real/manual viewport qualification is required before freezing I6.6D. The operator should run a real offline reconstruction and capture at least one mid-run frame showing provisional sparse cameras/landmarks and one final frame showing the admitted offline sparse reconstruction. I6.6D is not frozen until that evidence is reviewed.
