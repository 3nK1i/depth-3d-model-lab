# Phase 9.4E — Interactive Gaussian Viewer

## Scope

Phase 9.4E turns an integrity-verified persisted Gaussian `scene.pt` into an interactive appearance layer inside the existing Reconstruction VisPy viewport.

The implementation is deliberately presentation-only:

- no gsplat/Graphdeco import in the production `depth` process;
- no training, optimization, DA3 rerun, or neural-geometry authority;
- canonical geometry stays loaded and remains the exclusive picking/measurement authority;
- the existing backend-authored static Gaussian preview remains available as the high-fidelity raster reference;
- persisted tensors are loaded on CPU using PyTorch `weights_only=True` after persistent-scene file-integrity verification;
- interactive display is bounded and deterministically sampled above the configured point limit.

## UX

The existing Gaussian visualization group now exposes `Interactive`, `Static preview`, and `Hide` actions. Interactive mode reuses the Reconstruction viewport's Orbit/Pan/Zoom/ruler controls. Ruler hits continue to resolve against canonical geometry, never Gaussian splats.

## Authority

Gaussian remains `appearance_only`. It does not establish canonical geometry, metric scale, measurement, QC/reference, or certified-metrology authority.

## Validation gates

- focused loader/integrity tests;
- PyQt UI activation/static-fallback tests;
- persistence→interactive integration smoke;
- Phase 8 static Gaussian/representation-switching backward tests;
- Phase 9.4D persistence backward tests;
- pre-9.4E interactive measurement backward tests;
- real 9.4D South Building persisted scene check;
- manual Reconstruction Orbit/Pan/Zoom + ruler check while Gaussian appearance is visible.

Full application regression remains deferred to Phase 9.4H closure.
