# I4.2E — TempleRing Thin-Structure / Detail-Preservation Qualification

## Purpose

I4.2E tests whether the already-promoted room-scale `d12_sparse` manifold
predicate is safe when transferred to object-scale TempleRing detail. It does
not modify production filtering and does not authorize I4.3.

The controlled source is the successful I4.2D raw COLMAP MVS result. I4.2D
contains 47/47 geometric depth/normal maps and a raw geometric Stereo-Fusion
cloud, while preserving the I4.2B authoritative sparse model.

## Native dense-support provenance

I4.2D persisted the raw `fused.ply` using PyCOLMAP/COLMAP 4.1.1 with PLY output.
That same COLMAP call also writes `fused.ply.vis`, which stores the native
per-fused-point MVS visibility vectors in the same point ordering as the PLY.

I4.2E consumes this existing sidecar directly. It does **not** rerun Stereo
Fusion, PatchMatch, SfM, BA, undistortion, or pose refinement.

The native visibility is accepted only if its point count exactly matches the
frozen I4.2D PLY, every fused point has non-empty support, indices are unique per
point and lie in `[0, 46]`, and the binary file has neither truncation nor trailing
bytes. Global coverage of all 47 indices is diagnostic rather than required, because
visibility describes support of surviving fused points. The values are MVS
image indices, not COLMAP image IDs; I4.2E uses them only as per-view support
identities.

This corrected path supersedes the original replay/`Point3D.track` design from
I4.2E/I4.2E.1.

## Production-scale transfer evidence

The d12 predicate itself is reused unchanged through
`DenseManifoldRefinementPolicy` / `build_scene_relative_manifold_evidence`.

To avoid inventing a metric scale, the MVS support tracks are converted into a
scene-relative voxel representation using the established production-scale
fraction:

- voxel size = robust scene diagonal × `0.004`;
- source support = unique native fused-visibility image indices per voxel;
- 2+ view supported voxels are retained for the production-scale analogue;
- per-view voxel centroids provide dispersion evidence;
- the established production `DenseConfidenceFilterPolicy` classifier is
  reused directly with its locked defaults (including the support==2 Extended
  connectivity/strong-neighbour gates);
- the resulting Core/Extended analogue is used to rebuild the scene-relative
  d12 references.

This is deliberately labelled a **production-scale MVS transfer analogue**.
It does **not** claim to reproduce the complete upstream DA3/F evidence path.
The exact production d12 predicate is under test; the raw MVS source evidence
is reconstructed from COLMAP native fused-point visibility.

## Controlled counterfactual

I4.2E persists:

- `i4_2e_pre_d12_extended.ply` — MVS-derived Extended analogue before d12;
- `i4_2e_post_d12_extended.ply` — the same geometry after the exact d12
  predicate counterfactual;
- `i4_2e_d12_rejected_only.ply` — only d12-rejected voxels;
- `i4_2e_detail_analysis.npz` — aligned masks for reproducible preview;
- native `fused.ply.vis` provenance and SHA256 bindings.

The pre/post comparison uses the same coordinate frame. The preview uses the
same sampled indices and the same camera/framing on both sides; independent
auto-fit is forbidden.

## Descriptive detail-risk populations

Because no benchmark laser surface is distributed in this project, I4.2E does
not automatically label any rejected point as legitimate or illegitimate
detail. It reports descriptive populations only:

- Core-analogue rejection;
- 3+ view high-support rejection;
- supported line-like local-geometry proxy;
- supported sparse proxy;
- supported line-like + sparse proxy.

A `potential_detail_loss_signal` may be raised when d12 rejects Core or
supported line-like proxies. This is a review signal, not an automatic failure
or policy decision.

## Manual review contract

Manual review remains required for:

- thin vertical-member separation;
- small gaps/openings;
- edge definition;
- small protrusions;
- surface thickness / false bridging;
- local completeness vs noise;
- legitimate detail loss attributable to d12.

Four matched views are rendered by default: oblique, front, side, and top.
Each view includes a pre/post A/B plus a rejection overlay where d12-rejected
voxels are highlighted.

## Authority boundaries

I4.2E never:

- reruns SfM or BA;
- changes or rewrites camera poses;
- reruns PatchMatch;
- consumes held-out GT for filtering;
- applies d12 to the authoritative production scene;
- changes production filtering policy;
- promotes a dense provider or authorizes I4.3;
- claims metric scale, authoritative measurement, certified metrology, or a
  surface mesh.

## Success state

Execution success is `READY_FOR_MANUAL_REVIEW`, not an automatic detail winner.
I4.2F records the explicit final d12 transfer verdict and closes I4.2 only after
manual review of the matched previews.
