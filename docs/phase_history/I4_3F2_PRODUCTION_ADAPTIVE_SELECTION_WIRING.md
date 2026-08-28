# I4.3F2 — Production adaptive frame/view-selection wiring

## Scope

I4.3F2 wires the validated F1 adaptive-selection contract into the production offline path without selecting or promoting a dense provider.

Production changes are narrow:

- the application/offline-worker default `max_keyframes=80` is removed;
- `None` means adaptive planning with no permanent global cap;
- an explicit integer remains supported as an engineer override and is recorded as such;
- pre-SfM resource planning resolves a per-run target from available RAM/VRAM and finite source metadata;
- reduced finite image sets use deterministic first/last-preserving full-span sampling rather than prefix truncation;
- the offline worker persists `adaptive_pre_sfm_plan.json` and post-SfM `adaptive_dense_selection_ready.json`;
- a common immutable provider-input projection bridges selected registered views into all I4.3 provider adapters;
- the original `OfflineSfmResult`, camera transforms, sparse coordinates and workspace files remain unchanged;
- provider-local results can be rebound to original source/selected counts with adaptive-selection provenance;
- the MVS worker accepts an adaptively selected dense registered-view subset from the full authoritative sparse model while requiring selected poses and retained point coordinates to remain unchanged.

## Authority

PyCOLMAP remains the sole authoritative global camera/reconstruction frame. A provider-input projection is an immutable execution view, not a new reconstruction, trajectory, bundle adjustment or metric frame.

The following remain false:

- `provider_selected`
- `provider_promoted`
- `auto_policy_implemented`
- `hybrid_authorized`
- `metric_scale_claimed`
- `measurement_authoritative`
- `certified_metrology`
- `surface_mesh_generated`

`permanent_global_view_cap` remains `None`.

## Pre-SfM planning

The resource target uses the F1 transparent planning capacity and a documented F2 planning coefficient. It is a heuristic planning signal, not an empirical RAM/VRAM/runtime guarantee. Small sources that fit the current budget keep all frames. Larger sources may receive a run-specific target. More available resources increase the target; no constant value becomes a production cap.

Historical `max_keyframes` remains accepted only when explicitly supplied by an engineer/caller, preserving backward compatibility while removing it as the default.

## Post-SfM provider wiring

F2 does not implement Auto. Once an explicit provider is chosen by an authorized caller, `prepare_adaptive_provider_execution()`:

1. evaluates the F1 provider-specific plan against the authoritative registered views;
2. creates an immutable selected-view projection when reduction is required;
3. builds an explicit I4.3A selection request/decision for that concrete provider;
4. preserves source ordering for temporal providers;
5. refuses unavailable providers and authority drift.

After provider execution, `finalize_adaptive_provider_execution_result()` restores original source-count provenance and records selected count/reasons as adaptive selection.

## Validation

Local dependency-light validation covers F2 unit/integration/stress, F1 regression, ingestion/SfM workspace regression, MVS provider regression, Python compilation, shell syntax and hygiene. PyQt controller runtime tests are intentionally rerun on the authoritative `depth` environment because the model container does not include PyQt6.

## Next

I4.3F3 performs production validation/closure across small/large sources, constrained/high-resource profiles, provider families, temporal ordering, engineer overrides and authority invariants before I4.3G.
