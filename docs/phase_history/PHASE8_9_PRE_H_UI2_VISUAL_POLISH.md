# Phase 8.9 pre-H — UI-2 responsiveness and reconstruction visual polish

## Scope

Bounded presentation-only correction before the final post-fix qualification.
No reconstruction authority, SLAM pose, persistence, measurement/QC, Gaussian/NeRF,
or RTSP lifecycle contract is changed.

## Changes

- Live 3D source/runtime controls can negotiate reduced window widths instead of
  forcing large minimum widths that clip neighbouring controls.
- Snapshot action controls can shrink within the existing top row.
- The shared Live 3D/Snapshot render card enters a compact presentation when the
  viewport is short/narrow, keeping every available control inside the canvas.
- Reconstruction orientation gizmo keeps its 84x84 screen-space contract but no
  longer paints a visible container; the triad is painted slightly lower to align
  visually with `VIEWER Z-UP • ARBITRARY GAUGE`.
- The red current-camera frustum is preserved because it is valid pose evidence.
  Its symbolic depth is now bounded by the existing presentation-only adaptive
  scene scale instead of a fixed 0.35 reconstruction-unit depth, preventing it
  from dominating small arbitrary-gauge scenes as a large rotated rectangle.

## Explicitly unchanged

- No fullscreen behavior is added here; that belongs to UI-3.
- No Reconstruction toolbar semantics are added/removed here; UI-3 verifies and
  completes those interactions.
- Canonical geometry remains authoritative.
- Gaussian/NeRF remain appearance-only.
- `certified_metrology=false` unless separately qualified.

## Manual acceptance

Validate at minimum:

1. Live 3D maximized and windowed around 1366x768: source/runtime controls remain readable and bounded.
2. Snapshot maximized and windowed: top actions and floating render card stay inside their panels.
3. Reconstruction: orientation triad has no visible container and sits visually level with the bottom-left safety badge.
4. Reconstruction live/synthetic pose: current-camera frustum remains visible but restrained; trajectory/keyframes/landmarks remain unchanged.


## UI-2 correction pass

Real windowed validation found four bounded presentation defects after the first UI-2 overlay:

- the Reconstruction orientation gizmo referenced `PALETTE` without importing it, causing repeated `NameError` exceptions during `paintEvent`;
- the Live 3D source-type text was allowed to truncate instead of switching deliberately to an icon-only compact presentation;
- the idle Reconstruction centre diamond was a literal placeholder glyph (`QLabel("◇")`), not reconstruction geometry, so it is removed while the explanatory text remains;
- Stable Video's single control row needed compact copy at reduced desktop widths.

The correction remains UI-only. It does not alter source selection semantics, capture, inference, SLAM, canonical geometry, authority, persistence, measurement/QC, or isolated appearance backends. The gizmo painter is also exception-safe so a future paint error cannot leave an active `QPainter`.

## UI-2 final responsive correction

Manual compact-window validation found two remaining presentation issues:

- the shared 3D viewport's vertical tool strip could collide with the top-left
  `3D VIEWPORT` header and bottom-left orientation gizmo when the viewport was
  short (Live 3D, Snapshot, and Stable Video share this viewport);
- Stable Video runtime actions could collapse to effectively zero width because
  their horizontal size policy allowed the single control row to sacrifice them.

The shared viewport now reflows its tool strip horizontally beneath the header
whenever the existing compact viewport condition is active. This preserves every
viewport action while reserving independent top-left and bottom-left safe zones
for the header and orientation gizmo. Normal/tall viewports retain the existing
vertical tool strip.

Stable Video now keeps Start/Pause/Stop/Open actions non-collapsing. In compact
mode those actions use fixed icon-sized controls with tooltips; in normal mode
full labels return. Run-state semantics and enabled/disabled behavior are
unchanged.

## UI-2 final viewport containment refinement

Final manual acceptance identified two shared-viewport presentation details:

- the normal vertical tool strip was vertically centred, which could let its
  lower controls visually meet the bottom-left XYZ gizmo in Stable Video even
  though the viewport was tall enough to stay out of compact mode;
- Qt stylesheet border radii do not clip child widgets, so the native VisPy
  surface could still paint into the rectangular corner pixels of the rounded
  viewport canvas and make the 3D scene/grid appear to escape its container.

Normal vertical tools are now pinned to the same top-left safe lane used by the
compact strip, directly below the viewport header. The compact/normal choice and
all tool semantics remain unchanged. The shared canvas surface also applies a
rounded QWidget mask on resize, clipping both the idle grid and attached native
VisPy widget to the visual canvas boundary. This affects Live 3D, Snapshot, and
Stable Video uniformly and does not alter camera, geometry, inference, or render
semantics.

### Final compact viewport micro-fix

- Preserves the normal 260 px shared 3D canvas minimum, but relaxes it to 180 px only in compact mode so Live 3D, Snapshot, and Stable Video remain visually contained in short windows.
- Moves the shared XYZ orientation indicator 6 px inward and upward to avoid edge-crop risk.
- Reduces only the compact Render Mode card's vertical chrome and keeps the Reset action at a fixed 24 px height so it remains fully visible.
- No capture, inference, SLAM, reconstruction-authority, export, or persistence semantics are changed.

- Final gizmo nudge: moved the shared XYZ orientation overlay 6 px farther inward and 10 px upward from the already-corrected position, improving Measure & Inspect edge clearance without changing gizmo size, interaction, or rendering semantics.
