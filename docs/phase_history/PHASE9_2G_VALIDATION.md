# Phase 9.2G — Reviewed Masks → Measure & Inspect Integration

## Scope

Phase 9.2G connects accepted Phase 9.2 Auto Segment workspace masks to the existing
Measure & Inspect analysis path without changing depth, calibration, metric-scale,
measurement, QC, or canonical-geometry authority.

When accepted reviewed masks are available, Measure & Inspect exposes an explicit
`Use reviewed masks for Analyze Frame` choice. It is enabled by default for that
exact reviewed frame. In this mode the isolated measurement job skips perception
model inference and normalizes accepted semantic masks into the existing
`PerceptionFrameResult/ObjectObservation` contracts. When the choice is disabled,
or no reviewed selection is available, the legacy YOLO/perception path remains
unchanged.

## Safety / provenance

- The reviewed handoff is session-local and bound to the exact source BGR frame by SHA-256.
- A frame mismatch fails closed; reviewed masks are never silently applied to another image.
- Accepted object count must fit the existing Measure & Inspect object limit; no reviewed object is silently truncated.
- Instance names and semantic source/review provenance are preserved in the inspection bundle.
- If camera distortion rectification is active, reviewed masks are remapped with the same calibration mapping as RGB before alignment/3D extraction.
- `reviewed-auto-segment` is reported as the actual perception/semantic source when reviewed masks are used.
- Reviewed masks remain object-membership/semantic evidence only. They do not establish geometry, metric scale, measurement, QC, or metrology authority.
- Workspace persistence remains deferred to Phase 9.2H.

## Validation gates

- `scripts/run_phase9_2g_tests.sh quick`
- `scripts/run_phase9_2g_tests.sh integration`
- `scripts/run_phase9_2g_tests.sh backward`
- `python scripts/run_phase9_2g_reviewed_measurement_smoke.py`

The decisive real UI check is:

1. Capture/open an image in Measure & Inspect.
2. Open Auto Segment, accept at least one named object, and press **Done**.
3. Verify the Semantic Workspace card shows the accepted reviewed count and that
   `Use reviewed masks for Analyze Frame` is checked.
4. Click **Analyze Frame**.
5. Verify the result reports `Reviewed Auto Segment` as the perception source and
   preserves the accepted instance name(s), while depth/calibration/uncertainty
   behavior remains unchanged.
6. Uncheck reviewed-mask use and verify the selected YOLO/perception provider is
   available again as the explicit fallback.
