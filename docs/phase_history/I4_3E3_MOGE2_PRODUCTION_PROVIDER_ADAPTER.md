# I4.3E3 — MoGe-2 Production Single-Image Depth Provider Adapter

## Scope

I4.3E3 binds the application's existing MoGe-2 runtime to the canonical I4.3A offline dense-provider architecture. It does not change the interactive MoGe adapter, select a provider winner, implement Auto policy, or alter the authoritative reconstruction frame.

## Canonical authority

- Provider family: `single_image_depth`
- Implementation: `moge_2`
- Provider ID: `moge-2-single-image`
- Qualification state: `candidate`
- Authoritative camera/scene frame: PyCOLMAP SfM only
- Frame binding: `align_to_authoritative_sfm`
- All registered SfM views remain available by default
- Permanent global view cap: `None`
- Adaptive selection remains deferred to I4.3F

MoGe-2 native outputs are deliberately separated from canonical authority:

- native metric depth: retained as model evidence, then robustly scale-aligned per view into the arbitrary SfM gauge;
- model-predicted intrinsics: retained as evidence only and never used for canonical projection;
- direct model points: retained as evidence only and never inserted directly into canonical geometry;
- native normals: retained as evidence only and not used as canonical geometry authority;
- validity mask: consumed as a depth-validity gate;
- confidence: MoGe-2 exposes no native confidence map, so E3 does not fabricate one.

Canonical projection always uses the existing bundle-adjusted PyCOLMAP camera poses and camera model. Native metric depth does **not** establish physical reconstruction scale.

## Runtime provenance

The existing project installs the official MoGe package from `microsoft/MoGe` through the model requirements. E3 does not silently rewrite that established dependency path. The execution-free binding probe instead:

1. reads the installed `moge` distribution's PEP 610 `direct_url.json`;
2. requires an exact Git commit ID from the official `microsoft/MoGe` repository;
3. verifies `moge.model.v2.MoGeModel` resolves inside that installed distribution;
4. selects the existing hardware-adaptive MoGe variant (`ViT-S`, `ViT-B`, or `ViT-L`);
5. requires the selected Hugging Face checkpoint to already exist in the local cache with a resolved snapshot revision;
6. records both the source commit and checkpoint revision in binding evidence;
7. forces `local_files_only=True` for the offline provider run so network-side checkpoint updates cannot change evidence during execution.

The current `requirements/models-moge2.txt` Git URL itself is not commit-pinned. E3 therefore records the exact installed commit but does not claim that a future fresh environment is reproducible from that requirements line alone. Release-level dependency freezing remains an explicit packaging/hardening concern rather than being hidden inside this adapter step.

## Qualification boundaries

E3 preserves:

- `provider_promoted = false`
- `metric_scale_claimed = false`
- `measurement_authoritative = false`
- `certified_metrology = false`
- `surface_mesh_generated = false`
- cross-provider qualification pending I4.3G
- no universal provider winner

The project registry's MoGe license state remains `review_required`; E3 does not reinterpret that as commercial clearance.

## Automated validation

The E3 gate covers:

- immediate I4.3E2 predecessor sanity;
- existing MoGe adapter/output/registry/GPU-profile contracts;
- explicit-provider and no-fallback enforcement;
- all-view/no-global-cap behavior;
- source-image and SfM authority immutability;
- native metric-depth semantics;
- model-predicted-intrinsics semantics;
- direct-points/normals evidence preservation without authority promotion;
- no synthetic confidence;
- checkpoint/source revision drift rejection;
- runtime repository mismatch rejection;
- single-SfM-camera-model fusion guard;
- E1/E2 predecessor closure/authority checks;
- worker lazy-import/CUDA isolation.

Authoritative runtime binding is still performed on the workstation before E3 closure is declared.
