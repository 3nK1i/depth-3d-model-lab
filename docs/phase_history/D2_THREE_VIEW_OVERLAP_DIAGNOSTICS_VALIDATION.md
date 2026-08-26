# D2 — Three-View Overlap & Geometric-Consistency Diagnostics

## Purpose

D2 validates whether the three independently aligned/projected D1 dense RGB views agree where they observe the same scene geometry. It consumes the successful D1 artifacts and does **not** rerun PyCOLMAP or DA3.

D2 is intentionally diagnostic only:

- no spatial merge;
- no voxel fusion;
- no multi-view filtering mutation;
- no persistence;
- no metric-scale claim;
- no measurement/canonical-authority promotion.

## Primary evidence: bidirectional reprojection depth consistency

For every pair (0↔1, 1↔2, 0↔2), D2 transforms source-view canonical points into the other SfM camera, projects through the actual SfM camera model, and compares the projected source Z against the target view's retained aligned dense depth.

Points that lie behind a nearer target surface beyond the occlusion margin are classified as **occluded** instead of being counted as registration failures. Remaining comparable samples are classified as:

- consistent;
- front-surface conflict;
- moderate back-surface conflict.

The 0↔1 and 1↔2 adjacent pairs are the D2 acceptance gate. The 0↔2 endpoint pair is informational because temporal-third keyframes may have limited direct field-of-view overlap.

A bounded spatial nearest-neighbour diagnostic is stored as secondary evidence only; it is not the primary gate because non-overlapping fields of view can legitimately produce large nearest-neighbour distances.

## Default gate

For each direction of each required adjacent pair:

- at least 1,000 comparable samples;
- at least 50% consistent samples at ≤12% relative depth disagreement;
- median absolute relative depth residual ≤10%;
- P90 absolute relative depth residual ≤25%;
- front-surface conflict ratio ≤25%;
- source points >20% behind a nearer target depth are treated as occluded rather than failures.

These are arbitrary-gauge relative consistency diagnostics, not physical-accuracy limits.

## Expected artifacts

`/tmp/eia_d2_overlap_diagnostics/`

- `d2_summary.json`
- `d2_pair_01_diagnostic.npz`
- `d2_pair_12_diagnostic.npz`
- `d2_pair_02_diagnostic.npz`

The preview script uses the existing `PointCloudCanvas` and can color each pair by residual class or by source direction.

## Acceptance

D2 is green only when both required adjacent pairs pass bidirectional consistency. A degraded or inconclusive result stops progression to D3 so the disagreement can be diagnosed while only three views are involved.
