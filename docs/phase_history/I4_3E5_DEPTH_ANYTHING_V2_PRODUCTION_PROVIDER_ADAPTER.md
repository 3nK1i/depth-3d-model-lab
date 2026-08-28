# I4.3E5 — Depth Anything V2 Production Single-Image Depth Provider Adapter

## Scope

I4.3E5 binds the application's existing Depth Anything V2 image backend to the canonical I4.3A offline dense-provider architecture. It does not change the interactive adapter, select a provider winner, implement Auto policy, or alter the authoritative reconstruction frame.

## Canonical authority

- Provider family: `single_image_depth`
- Implementation: `depth_anything_v2`
- Provider ID: `depth-anything-v2-single-image`
- Qualification state: `candidate`
- Authoritative camera/scene frame: PyCOLMAP SfM only
- Frame binding: `align_to_authoritative_sfm`
- All registered SfM views remain available by default
- Permanent global view cap: `None`
- Adaptive selection remains deferred to I4.3F

Depth Anything V2 is explicitly relative, not metric. The existing adapter exposes a monotonic relative-distance map derived from the model's inverse-depth/disparity output. I4.3E5 therefore does **not** reuse the scale-only alignment used by metric E1–E4 providers.

For each registered view, E5 fits one robust affine relation in inverse-depth space using sparse SfM observations:

`1 / z_sfm ≈ a * (1 / d_dav2) + b`

with positive slope and robust residual qualification. The fitted relation is then used to convert that view's relative-depth map into the existing arbitrary SfM reconstruction gauge before fusion.

This fit is reconstruction-gauge alignment only. It does not establish physical metric scale.

Depth Anything V2 native/model evidence remains separated from canonical authority:

- relative depth: retained as evidence, then inverse-depth-affine aligned to SfM;
- estimated/virtual-FOV intrinsics: retained as evidence only and never used for canonical projection;
- adapter-projected points: discarded explicitly; they are not native model point geometry and never become canonical geometry;
- validity mask: consumed as a depth-validity gate;
- confidence: none; E5 does not fabricate a confidence map;
- native direct points/normals: unavailable in the project contract.

Canonical projection always uses the existing bundle-adjusted PyCOLMAP camera poses and camera model.

## Runtime provenance

The official source checkout lives at `third_party/Depth-Anything-V2` and is accompanied by `.eia_depth_anything_v2_ready.json`, which records the exact resolved Git commit used on the machine. The installer currently requests `main`, so E5 records whether that installer reference itself is commit-pinned rather than claiming fresh-install reproducibility.

For the selected Small/Base/Large checkpoint, E5 records:

- official Hugging Face repository;
- declared downloader revision;
- whether that revision is a commit pin;
- exact local checkpoint path;
- exact local checkpoint size;
- exact local checkpoint SHA256.

The execution-free binding probe also verifies that the official `DepthAnythingV2` class resolves from the recorded local source checkout.

## Qualification boundaries

E5 preserves:

- `provider_promoted = false`
- `relative_depth_promoted_to_metric_scale = false`
- `metric_scale_claimed = false`
- `measurement_authoritative = false`
- `certified_metrology = false`
- `surface_mesh_generated = false`
- cross-provider qualification pending I4.3G
- no universal provider winner

The project registry's Depth Anything V2 license state remains `variant_dependent_review_required`; E5 does not reinterpret that as commercial clearance.

## Automated validation

The E5 gate covers:

- immediate I4.3E4 predecessor sanity;
- existing Depth Anything V2 adapter/installer/downloader/registry/GPU-profile contracts;
- explicit-provider and no-fallback enforcement;
- all-view/no-global-cap behavior;
- source-image and SfM authority immutability;
- relative-depth / inverse-depth representation semantics;
- robust inverse-depth affine alignment and wrong-ordering rejection;
- estimated-intrinsics evidence-only semantics;
- no-confidence semantics;
- rejection of fabricated native points/normals;
- explicit discard of adapter-projected points;
- source revision and checkpoint checksum drift rejection;
- single-SfM-camera-model fusion guard;
- E1/E2/E3/E4 predecessor closure/authority checks;
- worker lazy-import/CUDA isolation.

Authoritative runtime binding is performed on the workstation before E5 closure is declared.
