# Phase 7.5 Step 8 — Inspection Orientation Gizmo Correction

## Scope

This focused pre-commit correction replaces the inspection viewport's scene-bound `XYZAxis` visual with a compact, screen-space orientation overlay.

## Behaviour

- Fixed to the bottom-left corner of **Object Point Cloud & 3D Box**.
- Transparent and mouse-pass-through.
- Rotates with the active VisPy turntable camera.
- Does not move during pan or zoom.
- Does not participate in object fitting or scene bounds.
- Does not change point-cloud, 3D-box, measurement, mask-correction, status-badge, or analysis behaviour.
- Uses X red, Y green, and Z blue with compact labels and arrowheads.

## Files

- Replaced: `app/widgets/inspection_3d_canvas.py`
- Added: `core/geometry/orientation_overlay.py`
- Added: `tests/unit/test_orientation_overlay.py`
- Replaced: `scripts/run_phase7_5_step8_tests.sh`

## Validation performed in the delivery environment

- Python compilation: passed.
- Orientation projection tests: 5 passed.
- Architectural guard confirms the inspection canvas no longer creates `scene.visuals.XYZAxis`.

PyQt6 and VisPy are not installed in the delivery container, so the real viewport rendering must be checked in the configured `depth` environment.

## Target manual check

1. Open Measure & Inspect and analyze a frame.
2. Select an object.
3. Confirm the XYZ triad appears in the bottom-left corner of the 3D viewport.
4. Rotate the object view: the triad orientation must rotate.
5. Pan and zoom: the triad must stay fixed in the same corner.
6. Click **Fit** and **Reset View**: the triad must remain fixed and must not affect fitting.
7. Confirm no new Qt/VisPy warnings appear during shutdown.
