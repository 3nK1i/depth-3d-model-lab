# I4.1.9 — Visual Reconstruction Acceptance & Geometry Authority Separation

## Purpose

I4.1.9 separates **visual reconstruction qualification** from the existing strict
**geometric/metric qualification**. It is additive: the I4.1.7/I4.1.8 dense
pair-consistency and orbit gates remain unchanged and continue to fail closed.

The visual milestone asks a narrower question: can the already reconstructed,
alignment-qualified observations produce a coherent persistent Dense RGB scene
that passes the unchanged fusion, filtering, persistence, and I2 visual/spatial
quality kernels?

## Why this step exists

The clean Bonsai positive control established strong SfM and broad camera-orbit
evidence, while strict dense pair consistency reduced a much larger aligned
observation pool to only four mutually strict-qualified views. The strict result
therefore remains useful as a geometry-confidence signal, but it should not by
itself prevent evaluation of the complete visual reconstruction.

I4.1.9 does **not** make the strict gates easier. It creates a separate visual
acceptance path over already alignment-qualified observations.

## Visual plan gates

A candidate dense plan is eligible for visual binding only when all are true:

- the candidate plan preserves full source span;
- alignment acceptance fraction is at least `0.80`;
- at least `10` observations are alignment-qualified;
- aligned observations preserve at least `0.80` of source span;
- all observations remain in the single authoritative PyCOLMAP SfM frame.

The visual selector deliberately does not use dense pair-PASS fraction as an
authorization gate. Pair-consistency evidence remains recorded as a separate
strict geometry diagnostic.

The final visual set is deterministically capped by the already-preserved fusion
cap (normally 40) while preserving the endpoints of the aligned observation
pool.

## Reused production kernels

I4.1.9 does not rerun PyCOLMAP, DA3, extra-frame localization, or dense-pair
analysis. It reuses existing evidence and then executes the existing unchanged:

1. dense voxel fusion;
2. F confidence/artifact filtering;
3. G persistent Dense RGB geometry;
4. I2 automated visual/spatial quality qualification.

An automated PASS produces `READY_FOR_MANUAL_VISUAL_REVIEW`. It does not close
the milestone automatically. Final closure requires explicit manual visual
review of fixed-view RGB previews.

## Authority boundaries

Even when visual reconstruction passes:

- SfM remains the sole global geometry authority;
- coordinates remain arbitrary-gauge reconstruction units;
- strict geometry qualification may remain degraded/deferred;
- `metric_scale_claimed=False`;
- `measurement_authoritative=False`;
- `certified_metrology=False`;
- `surface_mesh_generated=False`;
- production sampling/default behavior is unchanged;
- dense-consistency and orbit thresholds are unchanged.

Future camera calibration and ChArUco/reference evidence may refine intrinsics,
distortion, scale, and measurement authority without rewriting the provenance
of this visual reconstruction.

## Closure rule

I4.1 visual reconstruction closes only when:

- the visual plan gates pass;
- unchanged fusion/F/G/I2 automated gates pass;
- manual visual review explicitly passes.

Only after that manual closure is I4.2 thin-structure/detail-preservation work
authorized. Strict geometry/metric qualification remains a separate future
track when it has not already passed.
