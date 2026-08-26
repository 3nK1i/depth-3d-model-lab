# I2.5.1 — Spatial-Extremum Survivor Diagnosis

## Scope

I2.5.1 is a diagnostic-only follow-up to I2.5. It identifies the retained voxels that actually drive the pathological full-scene extent and explains why the six conservative I2.5 diversity policies failed to improve the full/robust envelope ratio.

It does **not** alter source geometry, formal F filtering, G persistence, SfM, DA3, metric/measurement authority, or mesh authority.

## Method

- Reconstruct the baseline Extended robust box using the same I2 2%/98% quantiles.
- Measure normalized excess beyond each of the six robust-box directions.
- Define a small directional extremum-driver population from the outer tail of each direction.
- Cross-reference those drivers with support, confidence, dispersion, topology, Core/peripheral/far labels, I2.4 parallax/baseline-range diversity, and exact source-view membership.
- Load all six I2.5 rejection masks and identify **stubborn extremum survivors** that are rejected by none of them.
- Decompose each I2.5 policy into full-diagonal shrink versus robust-diagonal shrink so ratio worsening cannot be misinterpreted.

## Authority boundaries

I2 closure, I3, production filtering, metric/measurement authority, and mesh reconstruction remain unauthorized by this diagnostic.
