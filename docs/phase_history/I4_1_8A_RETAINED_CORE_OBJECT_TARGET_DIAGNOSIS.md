# I4.1.8A — Retained-Core Object-Target Diagnosis

## Scope

Diagnosis-only follow-up to a completed I4.1.8 run whose production binding,
fusion, F filtering, G persistence, and I2 automated quality gates passed but
whose final retained-Core orbit target failed locked I4.1 coverage gates.

This step does **not** rerun or modify PyCOLMAP, DA3, dense A/B, fusion,
filtering, persistence, production binding, metric scale, measurement
authority, or mesh state.  It does not change any I4.1 threshold.

## Problem under test

I4.1.8 currently uses the coordinate-wise median of the entire retained Core
cloud as the final object-orbit target.  Retained Core is authoritative scene
geometry, but it may contain a dominant floor/support/background component and
therefore need not be an isolated object cloud.  The coordinate-wise median is
also not generally rigid-rotation equivariant.

I4.1.8A asks whether the saved whole-Core median is actually supported as an
object-centric target, rather than searching for any target that makes the
locked orbit gates pass.

## Evidence families

The diagnosis compares:

- saved/recomputed whole-Core coordinate-wise median;
- saved I4.1.7B trajectory-circle target;
- fresh production-bound trajectory-circle target;
- production optical-axis focus;
- a diagnosis-only retained-Core + camera-attention target candidate;
- target-independent bound-camera trajectory coverage;
- a deterministic dominant-plane probe over retained Core;
- target distances/local structure relative to that plane;
- coordinate-wise median drift under rigid-rotation probes.

The retained-Core + camera-attention candidate is derived only from retained
Core points.  When a strong dominant plane exists it is excluded first, then
remaining Core points are ranked by how consistently they lie near the bound
camera optical axes.  A rigid-transform-equivariant geometric median of the
highest-attention retained Core subset is reported as diagnosis evidence only.

## Fail-closed classification

A later I4.1.8B correction is recommended only when all expensive production
stages passed, the original I4.1.8 metric reproduces, no unrelated final gate
failed, and actual retained-Core + camera evidence supports a materially
different object target that satisfies the unchanged locked orbit geometry.

A passing trajectory-circle target by itself is **not enough**.  If it cannot
be supported by retained Core, the diagnosis remains fail-closed and does not
recommend target promotion.

## Authority

Always false in I4.1.8A:

- production target change;
- production sampling integration authorization;
- I4.1 final promotion;
- I4.2 detail-preservation authorization;
- threshold relaxation;
- metric scale claim;
- authoritative measurement;
- mesh generation.

## Manual preview

`run_i4_1_8a_target_preview.py` renders the retained Core, bound production
cameras, and diagnosis targets:

- magenta — current whole-Core median;
- yellow — production trajectory-circle center;
- green — retained-Core + camera-attention candidate, when available;
- orange — optical-axis focus.

The preview is diagnostic only.
