# I2.1 — Targeted Peripheral Artifact Diagnosis

## Scope

I2.1 diagnoses the connected peripheral/far geometry identified by I2 on the current full-video scene. It is deliberately **diagnostic only**.

It does not rerun SfM or DA3, does not refit or delete geometry, does not change Phase F thresholds, does not rewrite the Phase G persistent scene, and does not authorize mesh reconstruction.

## Evidence model

I2.1 reconstructs the exact per-view voxel membership for every retained Phase F Extended voxel using the same float32-aligned source reconstruction used by F. The reconstructed per-view observation matrix must exactly reproduce the retained `view_observation_counts`; mismatch is a hard error.

The phase then computes:

- leave-one-view support-collapse sensitivity for every selected view;
- automatic early/late suspect temporal segments from I2 peripheral attribution;
- optional custom segment sensitivity;
- support-2 peripheral/far co-observation pair rankings;
- directional collapse counts;
- support-only survivor spatial-span diagnostics;
- suspect-segment DA3↔SfM scale-profile context;
- local dense-consistency edge evidence touching suspect views.

A support-collapse counterfactual answers: *would this already-retained voxel cease to meet the ≥2-view evidence floor if the named view/segment were absent?* It does **not** claim to be an exact refusion/re-filtering result because point centroids, dispersion, connectivity and F classification are intentionally not recomputed in I2.1.

## Current-scene automatic segment policy

The selected trajectory is split around its median workspace order. Within each temporal half, the six views with the largest I2 peripheral-retained contribution are selected. For the current scene this is expected to recover the early and later suspect trajectory groups without hard-coding scene-specific orders.

## Authority

I2.1 preserves:

- arbitrary SfM gauge;
- `metric_scale_claimed=False`;
- `measurement_authoritative=False`;
- `surface_mesh_generated=False`;
- `geometry_changed=False`;
- `filter_policy_changed=False`.

I2.1 does not close I2 by itself. Its output decides whether an exact targeted counterfactual refusion is justified before I2 closure.
