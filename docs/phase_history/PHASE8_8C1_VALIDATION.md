# Phase 8.8C1 — Live Visualizer Foundation

Baseline: `phase8-reconstruction` at `11262e573b3ba8c86e5398e05497e645debe8cdd`.

## Scope

C1 replaces the Reconstruction viewport placeholder with a dedicated VisPy scene while preserving the existing Phase 8 reconstruction authority boundaries.

Implemented foundation:

- dedicated `ReconstructionCanvas` (not the Live 3D point-cloud buffers),
- viewer-frame conversion from authoritative `T_reconstruction_from_camera`,
- visible grid and XYZ axes,
- current-camera frustum + centre marker,
- bounded pose-history plumbing retained for C2 but trajectory rendering **disabled** in C1,
- Orbit / Pan / Zoom, Fit, Focus Camera, Reset, and Grid controls,
- 33 ms pose coalescing on the Qt/UI thread,
- explicit `VIEWER Z-UP • ARBITRARY GAUGE` safety label,
- producer-before-GL-surface shutdown ordering and late-observation guard.

## Authority / non-goals

C1 does not change reconstruction poses, scale, camera calibration contracts, or any reconstruction backend. It does not render sparse/dense/canonical geometry, a visible trajectory, Gaussian Splatting, NeRF, or DA3 output. No new metric or metrology claim is introduced.

`T_reconstruction_from_camera` remains authoritative. The display transform is presentation-only:

- viewer X = reconstruction X,
- viewer Y = reconstruction Z,
- viewer Z = -reconstruction Y.

Monocular live scale remains arbitrary and metrology remains uncertified unless later evidence establishes otherwise.

## Validation gates

1. `./scripts/run_phase8_8c1_tests.sh quick` — target: 15 focused tests.
2. `./scripts/run_phase8_8c1_tests.sh backward`
3. `python scripts/run_phase8_8c1_visualizer_preview.py` for visual inspection in a normal desktop session.
4. `python scripts/run_phase8_8c1_live_visualizer_smoke.py --max-frames 500 --playback-speed 2.0` for the actual MainWindow + Stella + real VisPy canvas path.
5. `python main.py` for a small normal-app sanity check.

The real visualizer smoke intentionally does **not** force Qt's offscreen platform. A C1 acceptance run must exercise a normal OpenGL-backed desktop surface; the older headless controller/status smokes remain useful but are not sufficient to validate this visualizer step.
