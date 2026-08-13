# Phase 8.9 pre-H UI-3 — Reconstruction controls + fullscreen

UI-3 is a bounded presentation/interaction polish step on the already-qualified Reconstruction workspace.
It does not change reconstruction authority, geometry, scale, runtime admission, persistence, export, measurement,
inspection/QC, or external backend execution.

## Scope

- Audit the existing floating Reconstruction viewport toolbar against the real `ReconstructionCanvas` behaviors.
- Keep data-conditional controls honest: semantic-object visibility remains disabled until semantic presentation exists.
- Add a dedicated fullscreen/focus control to the Reconstruction toolbar.
- Reuse the MainWindow focus-mode pattern already qualified by Live 3D; do not recreate the VisPy/OpenGL canvas.
- Escape and the same fullscreen button restore the complete Reconstruction workspace.
- Preserve the existing Live 3D Source/Depth/3D focus modes.

## Toolbar audit

The existing toolbar remains directly wired to real canvas/presentation behavior:

- Orbit / Pan / Zoom → interaction mode
- Fit → fit known scene/path
- Focus camera → current or selected offline camera
- Reset → reset reconstruction camera
- Grid → world-grid visibility
- Trajectory → live camera trajectory visibility
- Landmarks → sparse SLAM landmarks visibility
- Keyframes → historical keyframe-camera visibility
- Sparse envelope → offline sparse presentation envelope visibility
- Semantics → semantic-object context visibility, enabled only when semantic context exists
- Diagnostics → live tracking diagnostics visibility
- Fullscreen → UI-3 focus mode only

No toolbar action establishes metric scale or changes canonical, measurement, inspection/QC, or appearance authority.

## Fullscreen behavior

Fullscreen is presentation-only. MainWindow hides global chrome while `ReconstructionPage` hides its own header,
source/runtime cards, and settings sidebar. The existing Reconstruction canvas remains the same object and expands
inside the existing page. Restore is available through the same fullscreen button or Escape.

## Validation

Run:

```bash
./scripts/run_phase8_9_pre_h_ui3_tests.sh quick
./scripts/run_phase8_9_pre_h_ui3_tests.sh affected
```

Manual acceptance should check Reconstruction at 1366×768 and a maximized desktop window, including fullscreen enter,
Escape restore, toolbar visibility, orientation gizmo/safety label, and one return to Live 3D fullscreen.

## UI-3 correction pass

- Moved the idle Reconstruction copy slightly below the world-origin triad so the centered scene-axis overlay and text no longer intersect.
- Routed Escape from the focused Reconstruction VisPy native canvas back through the page to MainWindow focus-mode exit. This keeps Escape reliable after orbit/pan/zoom interaction transfers keyboard focus to the canvas.
- The correction is presentation/input-routing only; it does not recreate the GL surface or alter reconstruction geometry/runtime authority.

## UI-3 final idle-copy positioning polish

- Biased the idle Reconstruction headline/detail farther into the lower viewport so it remains comfortably separated from the world-origin XYZ axes at normal and fullscreen sizes.
- No world-axis geometry, zoom behavior, reconstruction data, camera pose, or authority semantics are changed by this adjustment.

## UI-3 final marker-aligned idle-copy nudge

- Rebalanced the idle-copy vertical spacer from 8:2 to 4:3 so the headline/detail sits just below the world-origin triad, matching the final visual target without intersecting the scene axes.
- This remains a presentation-only adjustment; no reconstruction geometry, camera behavior, runtime state, or authority semantics are changed.
