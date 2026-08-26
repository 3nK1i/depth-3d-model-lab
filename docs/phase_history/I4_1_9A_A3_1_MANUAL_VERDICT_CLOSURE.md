# I4.1.9A A3.1 — Manual Verdict Persistence & Closure

## Scope

A3.1 persists the explicit human verdict from the controlled House DA3-vs-COLMAP-MVS A/B. It is a closure/evidence step only. It does not rerun DA3, MVS, SfM, BA, undistortion, PatchMatch or Stereo Fusion, and it does not alter camera geometry.

The reviewed A3 evidence uses the same House input, same authoritative I4.1.7 PyCOLMAP SfM frame and shared camera/framing for both rendered branches. Independent auto-fit is forbidden.

## Manual House verdict

The accepted verdict is:

`mvs_better_but_imperfect`

COLMAP MVS is clearly more coherent than the current DA3 I4.1.9 reconstruction across matched oblique/front/side/top views. The MVS result shows substantially less radial/fan streaking and halo, cleaner silhouettes, less surface thickening/smearing, fewer detached/ghost artifacts, and stronger roof/wall structure.

MVS is not treated as perfect: residual point-cloud roughness/noise, local incompleteness and finite surface thickness remain. No mesh or metrology-qualified surface is claimed.

## Evidence integrity

Before closure, the finalizer re-hashes:

- every DA3/MVS/A-B render for all four views;
- the DA3 I4.1.9 summary;
- the DA3 I2 summary and spatial diagnostics NPZ;
- the A2 MVS summary and fused PLY;
- the frozen A1 camera-authority manifest;
- the shared I4.1.7 source summary.

Any mutation or missing evidence fails closed. The closure file is immutable and cannot overwrite an existing closure.

## Scientific conclusion

Because MVS is materially cleaner while using the same authoritative camera solution/frame, the House evidence supports the learned-depth/alignment/fusion branch as the primary location of the current DA3 failure. This does not claim that camera-model error is mathematically zero or completely exonerate every camera assumption.

MVS is supported as a strong static-object-orbit provider candidate. This step does **not** perform the production provider promotion/refactor; that remains I4.3.

## Authority

A3.1 preserves:

- one authoritative PyCOLMAP SfM frame;
- arbitrary SfM scale;
- `metric_scale_claimed=False`;
- `measurement_authoritative=False`;
- `certified_metrology=False`;
- `surface_mesh_generated=False`;
- `dense_provider_promoted=False`.

## Closure decision

A passing A3.1 closes the House MVS positive control (`I4.1.9A`) and authorizes only the next roadmap milestone: **I4.2 — Temple Ring thin-structure/detail-preservation evaluation**.

I4.3 provider architecture remains unauthorized until I4.2 is completed and reviewed.
