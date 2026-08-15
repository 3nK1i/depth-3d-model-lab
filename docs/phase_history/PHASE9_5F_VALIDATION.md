# Phase 9.5F — NeRF Controls & Scene Layers UX

Phase 9.5F extends the already-qualified Phase 9.5E interactive Nerfacto viewer with presentation-only controls and a first-class scene-layer toggle.

## Delivered

- Render detail profiles: Fast 320×180, Balanced 480×270, Detail 640×360.
- Detail 640×360 remains the default to preserve 9.5E behavior.
- Interactive image opacity 35–100% for canonical/appearance comparison.
- Automatic refresh after canonical navigation, enabled by default.
- Explicit `Refresh view` action when automatic refresh is disabled.
- `Reset display` restores the 9.5E defaults.
- `NeRF appearance` added to Scene Layers beside Gaussian appearance.
- Static Preview remains backend-free and always uses full image opacity.
- Ruler/picking remain canonical-only; NeRF stays `appearance_only`.

## Authority / isolation

No NeRF control modifies the persisted checkpoint/config/dataset, canonical geometry, metric scale, measurement/QC authority, or certified-metrology state. Nerfstudio/tiny-cuda-nn remain isolated in the existing 9.5E external worker. 9.5F does not retrain or benchmark the backend.

## Validation

Run `scripts/run_phase9_5f_tests.sh quick|integration|backward`, then validate the real persistent scene with `scripts/run_phase9_5f_real_nerf_controls_check.py`. Manual UI validation covers the new controls/layer toggle plus a small ruler/Preview/Hide compatibility sanity check.
