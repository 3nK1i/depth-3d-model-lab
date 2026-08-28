# I4.3E4 — Depth Pro Production Single-Image Depth Provider Adapter

## Scope

I4.3E4 binds the application's existing Apple Depth Pro runtime to the canonical I4.3A offline dense-provider architecture. It does not change the interactive Depth Pro adapter, select a provider winner, implement Auto policy, or alter the authoritative reconstruction frame.

## Canonical authority

- Provider family: `single_image_depth`
- Implementation: `depth_pro`
- Provider ID: `depth-pro-single-image`
- Qualification state: `candidate`
- Authoritative camera/scene frame: PyCOLMAP SfM only
- Frame binding: `align_to_authoritative_sfm`
- All registered SfM views remain available by default
- Permanent global view cap: `None`
- Adaptive selection remains deferred to I4.3F

Depth Pro native outputs are separated from canonical authority:

- native metric depth: retained as model evidence, then robustly scale-aligned per view into the arbitrary SfM gauge;
- focal-length-derived/model-predicted intrinsics: retained as evidence only and never used for canonical projection;
- estimated intrinsics fallback: preserved through the existing registry normalization when model focal length is unavailable;
- validity mask: consumed as a depth-validity gate;
- confidence: none; E4 does not fabricate a confidence map;
- direct points/normals: not native Depth Pro outputs in the existing project contract, and E4 rejects any worker result that fabricates them.

Canonical projection always uses the existing bundle-adjusted PyCOLMAP camera poses and camera model. Native metric depth does **not** establish physical reconstruction scale.

## Runtime provenance

The existing model requirement installs the official Apple `ml-depth-pro` package from Git and uses the local `checkpoints/depth_pro.pt` checkpoint. E4 does not silently rewrite either dependency.

The execution-free binding probe instead:

1. reads the installed `depth-pro` distribution's PEP 610 `direct_url.json`;
2. requires an exact resolved Git commit ID from the official `apple/ml-depth-pro` repository;
3. verifies `create_model_and_transforms` resolves inside that installed distribution;
4. verifies the local checkpoint exists above the same minimum-size threshold used by the existing downloader;
5. records the exact checkpoint size and SHA256 in binding evidence;
6. records whether `requirements/models-depth-pro.txt` is itself commit-pinned.

The current requirement Git URL is not commit-pinned, so E4 records `requirement_commit_pinned=false` rather than claiming fresh-install reproducibility. The exact installed VCS commit and local checkpoint SHA256 remain auditable evidence for this validated machine state.

## Qualification boundaries

E4 preserves:

- `provider_promoted = false`
- `metric_scale_claimed = false`
- `measurement_authoritative = false`
- `certified_metrology = false`
- `surface_mesh_generated = false`
- cross-provider qualification pending I4.3G
- no universal provider winner

The project registry's Apple Depth Pro license state remains `review_required`; E4 does not reinterpret that as commercial clearance.

## Automated validation

The E4 gate covers:

- immediate I4.3E3 predecessor sanity;
- existing Depth Pro adapter/checkpoint/registry/GPU-profile contracts;
- explicit-provider and no-fallback enforcement;
- all-view/no-global-cap behavior;
- source-image and SfM authority immutability;
- native metric-depth semantics;
- model-predicted/estimated-fallback intrinsics semantics;
- no-confidence semantics;
- rejection of fabricated direct points/normals;
- installed source revision and checkpoint checksum drift rejection;
- runtime distribution provenance checks;
- single-SfM-camera-model fusion guard;
- E1/E2/E3 predecessor closure/authority checks;
- worker lazy-import/CUDA isolation.

Authoritative runtime binding is performed on the workstation before E4 closure is declared.
