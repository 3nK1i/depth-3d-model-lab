# I4.1 — Object / Connector Orbit Qualification

## Purpose

I4.1 starts the object-scale reconstruction campaign after the closed I3 second-scene validation. It verifies that the existing offline Dense RGB pipeline can reconstruct a **small static object** from a camera orbit and that the selected qualified views actually surround the object with useful viewpoint diversity.

This step is intentionally **not** a mesh step and does not change the production filtering policy. It establishes capture/SfM/dense/orbit evidence before I4.2 stresses thin connector/detail preservation.

## Capture model

- The object must remain static in the world.
- The camera moves around the object.
- A turntable capture with a static camera and rotating object is not supported by the current global-SfM world model.
- Slow motion, strong overlap, stable focus/exposure and a textured/non-reflective object/background are preferred.
- Exhaustive matching is the I4 default because an orbit revisits/opposes viewpoints and may contain loop closure.

## Pipeline

1. Require the closed I3 authorization (`i4_connector_orbit_authorized=True`).
2. Run/reuse I1 end-to-end offline Dense RGB reconstruction.
3. Run the unchanged I2 baseline-quality qualification.
4. Estimate a robust object/scene target from retained Core geometry.
5. Extract all registered SfM camera poses and the exact selected qualified camera poses.
6. Measure scale-invariant orbit coverage:
   - azimuth coverage around target;
   - occupied azimuth bins;
   - opposite-side viewpoint angle;
   - target-facing fraction;
   - standoff consistency;
   - orbit-plane residuals and sequence travel as diagnostics.
7. Require every automated orbit gate before manual review.

## Default automated gates

- ordinary I2 baseline quality: PASS
- registered fraction >= 0.80
- dense alignment fraction of registered views >= 0.80
- selected qualified views >= 10
- selected azimuth coverage >= 150 degrees
- selected occupied bins >= 5 / 12
- selected opposed-view angle >= 120 degrees
- >= 75% of selected cameras face the robust target within 60 degrees
- selected standoff p90/p10 <= 2.50

The thresholds are capture-qualification gates. They must not be relaxed simply to make a difficult video pass.

## Authority boundaries

I4.1 preserves:

- arbitrary monocular SfM gauge;
- no metre claim;
- no authoritative measurement;
- no certified metrology;
- no surface mesh;
- no new filter policy;
- no I4.2 authorization until manual orbit review passes.

## Expected success state

`READY_FOR_MANUAL_ORBIT_REVIEW`

Then inspect Dense RGB together with the selected camera orbit. The camera trajectory should visibly surround the intended object rather than merely traverse one side of a room/background.

## Next

I4.2 — Thin Structure / Connector Detail Preservation Diagnosis.

I4.2 will specifically test whether the manifold cleanup that was safe on room-scale scenes can preserve legitimate thin connector pins, edges, slots and small protrusions. It must not assume that room-scale artifact logic is automatically safe for object-scale detail.
