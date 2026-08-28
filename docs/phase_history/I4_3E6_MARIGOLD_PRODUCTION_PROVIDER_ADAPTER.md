# I4.3E6 — Marigold Production Single-Image Depth Provider Adapter

## Scope

I4.3E6 binds the existing Marigold Depth v1.1 image backend to the canonical I4.3A offline dense-provider architecture. It does not change the interactive adapter, select a provider winner, implement Auto policy, or alter the authoritative reconstruction frame.

## Canonical authority

- Provider family: `single_image_depth`
- Implementation: `marigold`
- Provider ID: `marigold-single-image`
- Qualification state: `candidate`
- Authoritative camera/scene frame: PyCOLMAP SfM only
- Frame binding: `align_to_authoritative_sfm`
- All registered SfM views remain available by default
- Permanent global view cap: `None`
- Adaptive selection remains deferred to I4.3F

Marigold is affine-invariant relative depth, not a physical metric provider. Its official pipeline can report either `depth` or `disparity`; I4.3E6 aligns in the matching affine domain:

- `prediction_type=depth`: `z_sfm ≈ a * d_marigold + b`
- `prediction_type=disparity`: `1 / z_sfm ≈ a * (1 / d_adapter) + b`

The mature app adapter converts native disparity to monotonic relative distance, so its reciprocal restores the affine disparity domain for the second case. Both fits are per-view reconstruction-gauge alignment only. Neither establishes physical metric scale.

Marigold model evidence remains separate from canonical authority:

- relative depth: retained, robustly aligned to SfM, then projected with PyCOLMAP cameras;
- estimated virtual-FOV intrinsics: retained as evidence only;
- adapter-projected points: discarded explicitly and never canonical;
- validity mask: consumed as a depth-validity gate;
- optional ensemble uncertainty: preserved as lower-is-better evidence and transformed to fusion confidence as `1/(1+uncertainty)` only when present;
- fast/balanced profiles without uncertainty: no confidence is fabricated;
- native direct points/normals: unavailable in the project contract.

## Runtime provenance

Marigold uses the official Diffusers `MarigoldDepthPipeline` plus the locally downloaded Hugging Face snapshot `prs-eth/marigold-depth-v1-1`.

The binding probe records and verifies:

- installed Diffusers distribution name/version;
- imported `MarigoldDepthPipeline` resolves to that installed distribution;
- whether the project's Diffusers requirement is exact-version pinned;
- local snapshot path;
- `.ready` marker repository/requested revision/resolved revision;
- resolved snapshot revision is an exact 40-character commit;
- `.ready` and `model_index.json` SHA256;
- snapshot file count and total size;
- CUDA/GPU availability;
- automatically resolved Marigold profile (`efficient`, `balanced`, or `ultra`).

The downloader already resolves `main` to a concrete Hugging Face commit before downloading and persists that resolved revision in `.ready`. The Diffusers requirement itself is version-ranged, so I4.3E6 reports that fact instead of making a false fresh-install reproducibility claim.

## Qualification boundaries

I4.3E6 preserves:

- `provider_promoted = false`
- `relative_depth_promoted_to_metric_scale = false`
- `metric_scale_claimed = false`
- `measurement_authoritative = false`
- `certified_metrology = false`
- `surface_mesh_generated = false`
- cross-provider qualification pending I4.3G
- no universal provider winner

The project registry's Marigold license state remains `review_required`; I4.3E6 does not reinterpret it as commercial clearance.

When E6 binding passes, all six I4.3E single-image provider adapters are closed at the integration/binding level. They remain candidates until later controlled cross-provider qualification. The next roadmap step is I4.3F adaptive frame/view selection.

## Automated validation

The E6 gate covers:

- immediate I4.3E5 predecessor sanity;
- existing Marigold adapter/downloader/installer/registry/GPU-profile contracts;
- both depth-domain and disparity-domain affine SfM alignment;
- optional uncertainty and no-uncertainty behavior;
- explicit-provider/no-fallback enforcement;
- all-view/no-global-cap behavior;
- source-image and SfM authority immutability;
- estimated-intrinsics evidence-only semantics;
- rejection of fabricated native points/normals;
- explicit discard of adapter-projected points;
- runtime/checkpoint revision drift rejection;
- single-SfM-camera-model fusion guard;
- E1–E5 predecessor closure/authority checks;
- worker lazy-import/CUDA isolation;
- syntax/compile/filesystem hygiene.

Authoritative runtime binding is performed on the workstation before I4.3E6 closure is declared.
