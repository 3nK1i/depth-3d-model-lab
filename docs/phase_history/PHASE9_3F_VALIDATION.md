# Phase 9.3F Validation — Professional Scene Layers & Visibility Controls

Phase 9.3F adds presentation-only visibility controls to the existing Reconstruction/Digital Twin viewer. It does not change reconstruction, persistence, semantic, measurement, QC, calibration, scale, or backend authority.

## Scope

- Add one collapsible `SCENE LAYERS` group in the existing Reconstruction settings sidebar.
- Preserve the existing viewport toolbar as quick access and mirror its grid, camera-frustum, trajectory, sparse-landmark, sparse-envelope and semantic-master toggles.
- Add independent presentation controls for the canonical surface, world axes, semantic labels, selected canonical-surface highlight and qualified inspection evidence.
- Keep visibility preferences session-local and preserve them across persistent-scene reloads.
- Keep unavailable evidence-backed layers disabled rather than fabricating resources.
- Preserve the 1366×768 responsive sidebar and existing horizontal-scroll prohibition.

## Authority invariants

- Hiding the canonical surface never unloads or demotes canonical geometry.
- Camera/frustum/trajectory controls remain supporting presentation context only.
- Semantic support, labels and selection remain semantic context only.
- Inspection evidence remains the existing source-authored E3/E4 result; hiding it does not alter validity, uncertainty, rejection or QC state.
- `certified_metrology` remains false unless separately qualified.
- No metric scale, camera pose, semantic geometry, measurement geometry or QC decision is fabricated by a layer toggle.
- No YOLO/SAM2, reconstruction, DA3, Gaussian or NeRF backend is started by persistent-scene layer presentation.

## Automated validation

Run:

```bash
./scripts/run_phase9_3f_tests.sh quick
./scripts/run_phase9_3f_tests.sh integration
./scripts/run_phase9_3f_tests.sh backward
```

The dependency-light scene-layer smoke is:

```bash
python scripts/run_phase9_3f_scene_layers_smoke.py
```

## Manual validation

Run the real PyQt/VisPy preview:

```bash
python scripts/run_phase9_3f_scene_layers_preview.py --show
```

At 1366×768 verify:

1. `SCENE LAYERS` is readable, compact and horizontally unclipped.
2. Canonical surface, world grid/axes, camera frustums, camera trajectory, sparse context and semantic layers can be hidden/restored cleanly.
3. Existing toolbar quick toggles stay synchronized with their corresponding layer switches.
4. Semantic labels, selected-surface highlight and inspection evidence can be controlled independently under the semantic master visibility gate.
5. Turning a layer off never changes the sidebar's authority/scale/measurement/QC statements.
6. Scene reload restores materialized resources while preserving the current visibility preferences.
7. Orbit/Pan/Zoom/Fit/Reset and semantic Prev/Next/Focus/Clear continue to work.
8. Normal shutdown releases canonical mesh/camera/semantic/inspection viewer resources without a VisPy crash.

Phase 9.3G remains the next locked roadmap step; advanced reconstruction/dense-geometry provider UI is not part of 9.3F.
