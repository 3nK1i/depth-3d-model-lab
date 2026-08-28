# I4.3G2B — TempleRing matched-view diagnostic

This step adds a diagnostic-only visual A/B for the frozen TempleRing production-native MVS and DA3 artifacts.

## Locks

- No SfM, MVS, or DA3 rerun.
- PyCOLMAP remains the sole camera/global-frame authority.
- Both clouds must reference the identical frozen authoritative SfM SHA and frame id.
- The visual center and scale are derived only from the authoritative sparse SfM landmarks.
- MVS and DA3 use the same camera center, distance, FOV, azimuth, and elevation for each paired render.
- Independent provider auto-fit is forbidden.
- Review labels are blinded as Cloud A / Cloud B until the manual review is recorded.
- Point count is informational only and may not determine a winner.
- No provider selection, promotion, Auto policy, hybrid, metric-scale, measurement, metrology, or mesh authority is introduced.

## Review dimensions

The frozen G1 dimensions are preserved: global completeness, legitimate hole recovery, false geometry/floaters, surface noise/roughness, thin-structure preservation, edge definition, small-detail preservation, and false bridging/thickening.

Two shared framings are rendered (`detail`, `context`) across fixed `oblique`, `front`, `side`, and `top` views. The context framing exists to make detached or outlying geometry visible without fitting either provider independently.
