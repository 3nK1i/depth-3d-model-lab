# Phase 8.9 Pre-H UI-4 — Combined UI Qualification

UI-4 is a **qualification-only** checkpoint for the already committed UI-2 and
UI-3 work. No production application files are changed in this substep.

Baseline: `phase8-reconstruction` at
`8673908caa8fdacb2227f81fecfc778d665e327c`.

## Scope

The gate composes the previously accepted UI-2 and UI-3 behavior instead of
adding another visual feature:

- Live 3D, Snapshot and Stable Video compact/windowed responsiveness.
- Shared viewport tool, render-card, canvas containment and gizmo safe zones.
- Measure & Inspect / shared orientation-gizmo backward sanity through the UI-2 gate.
- Reconstruction toolbar wiring and visual-surface preservation.
- Reconstruction fullscreen/focus mode and Escape after VisPy takes keyboard focus.
- Existing Live 3D fullscreen behavior after returning from Reconstruction.
- Main-window shell, Phase 7 mode switching, shutdown lifecycle and bounded RTSP
  FPS plumbing sanity.

## Automated gates

```bash
./scripts/run_phase8_9_pre_h_ui4_tests.sh quick
./scripts/run_phase8_9_pre_h_ui4_tests.sh affected
```

`quick` validates the UI-4 composition contract. `affected` then runs the full
UI-2 affected gate, the full UI-3 affected gate, and the small shell/lifecycle
backward-compatibility sanity set.

## Manual acceptance

Launch once with the real process status preserved:

```bash
set -o pipefail
PYTHONFAULTHANDLER=1 python -X faulthandler main.py \
  2>&1 | tee /tmp/depth3d_pre_h_ui4_manual.log
rc=${PIPESTATUS[0]}
echo "REAL_EXIT_CODE=$rc"
```

During that single run:

1. At a normal/maximized size, visit Live 3D, Measure & Inspect, Stable Video,
   Snapshot and Reconstruction.
2. Resize to a compact/windowed layout and visually confirm Live 3D, Snapshot
   and Stable Video remain contained: no viewport bleed, no toolbar/gizmo
   collision, and the Render Mode Reset action remains visible.
3. Confirm the Measure & Inspect/shared XYZ gizmo keeps comfortable edge
   clearance.
4. In Reconstruction, enter fullscreen, orbit/rotate so the VisPy canvas owns
   keyboard focus, press Escape, and confirm the complete workspace restores.
5. Re-enter Reconstruction fullscreen and restore using the fullscreen button.
6. Return to Live 3D and verify its fullscreen/restore path still works.
7. Close the application normally.

Acceptance requires `REAL_EXIT_CODE=0` and no Python/Qt fatal markers:

```bash
grep -E \
  "NameError|QBackingStore::endPaint|QPaintDevice|Uncaught exception|Traceback|Fatal Python error" \
  /tmp/depth3d_pre_h_ui4_manual.log \
  || echo "PASS: UI-4 manual runtime clean"
```

## Exit criteria

UI-4 closes when both automated gates pass, the manual combined scenario is
visually accepted, the real application exit status is zero, and the Git diff
contains only the four UI-4 qualification files. After that, commit/push this
qualification checkpoint and proceed to the final post-fix whole-app
requalification before Phase 8.9H.

## Quick-gate fixture correction

The initial combined UI-4 test referenced a non-existent `MainWindow._canvas` attribute while checking the already-qualified Live 3D fullscreen path. Live 3D owns its persistent `PointCloudCanvas` through `MainWindow._inference_controller._point_cloud_canvas`; the qualification test now uses that established ownership path. No production code was changed.
