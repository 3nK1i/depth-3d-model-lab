# I4.1.6 — Additive Decoupled Dense-Candidate Localization Integration

## Purpose

I4.1.6 converts the I4.1.5 decoupled-sampling counterfactual into a real,
opt-in integration path while preserving the existing reconstruction
architecture. The global PyCOLMAP reconstruction remains the only geometry
frame. Additional local-density frames are observations only and must be
localized into that frame before they can participate in dense processing.

## I4.1.6A scope

The first validation stage performs real source-frame materialization and
2D–3D pose localization against landmarks from the existing SfM model. It:

- preserves I4.1.3 full-span global sampling;
- evaluates both `uniform_localized_dense_pool` and
  `global_anchors_plus_local_neighbors` by default;
- caps each dense candidate plan at 120 frames;
- reuses exact global-anchor poses when a dense candidate coincides with a
  registered SfM keyframe;
- localizes every other candidate from SIFT matches to existing SfM landmark
  observations, followed by robust PnP and reprojection gates;
- records candidate-localization provenance and rejects candidates that fail
  pose evidence.

It does **not** rerun PyCOLMAP, DA3, dense-pair qualification, fusion, filtering,
persistence, or any metric/measurement/mesh stage.

## Architectural invariants

- One authoritative SfM frame only.
- No second reconstruction/map.
- Extra dense frames are never accepted from timestamp interpolation alone.
- Existing dense-consistency thresholds remain unchanged.
- Existing production default remains unchanged during I4.1.6A.
- Arbitrary SfM gauge remains arbitrary; no metre claim is introduced.

## Gate to I4.1.6B

A local plan must preserve full source span, accept at least 10 extra localized
candidates, and localize at least 80% of its extra candidates. Passing this gate
authorizes only the next real DA3/dense-consistency A/B validation inside
I4.1.6. It does not authorize production promotion or I4.2.
