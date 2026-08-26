# I4.3C — DA3 Production Provider Adapter

## Scope

I4.3C binds the already-integrated isolated Depth Anything 3 (DA3) runtime behind the I4.3A canonical dense-provider contract as a **multi-view AI-depth** provider. The step is intentionally narrow: it does not redesign the DA3 runtime, implement Auto selection, add temporal/single-image providers, change production filtering, implement adaptive view selection, or promote DA3.

DA3 remains a **candidate** until the cross-provider operating-envelope qualification in I4.3G and architecture closure in I4.3H. Historical House DA3 evidence is retained as evidence, not treated as a universal quality claim.

## Locked authority

- PyCOLMAP offline SfM remains the sole authoritative camera/global-frame owner.
- I4.3C requires an existing authoritative `OfflineSfmResult`; it never reruns SfM, bundle adjustment, or pose refinement.
- DA3 native depth/confidence and predicted extrinsics are validated/preserved as provider evidence.
- DA3-predicted extrinsics are **never consumed for canonical projection** in I4.3C.
- Each accepted DA3 depth map is aligned into the existing arbitrary SfM gauge with the already-qualified SfM/dense alignment contract.
- Canonical projection/fusion always uses `T_reconstruction_from_camera` from the authoritative PyCOLMAP SfM result and the `offline_sfm_bundle_adjusted` pose layer.
- Input staged images plus authoritative SfM cameras/landmarks are integrity checked before/after provider execution.
- No metric scale, measurement authority, certified metrology, surface mesh, production filter-policy change, or provider promotion is established.

## Provider architecture

The adapter reuses the existing `DA3ExternalWorkerRunner` and isolated `scripts/da3_workers/da3_worker.py`; no second DA3 runtime or worker protocol is introduced.

The canonical path is:

1. resolve every registered authoritative SfM view;
2. verify staged-image hashes and SfM authority state;
3. execute one isolated DA3 multi-view request using the existing DA3 provider runtime;
4. validate model/source revisions and DA3 result arrays;
5. adapt DA3 depth/confidence pixel grids only when the resize is sufficiently near-uniform; fail closed on unsafe anisotropic resampling;
6. align each DA3 depth map to the authoritative SfM gauge using sparse SfM correspondences;
7. reject individual views whose depth/SfM alignment does not meet the qualified gate;
8. project/fuse accepted depth maps using only authoritative PyCOLMAP camera poses;
9. persist raw canonical pre-filter geometry as NPZ/PLY plus native DA3 artifacts/provenance;
10. verify SfM/image integrity and authority flags unchanged.

The existing qualified voxel-fusion kernel is reused. I4.3C does not invent a new fusion/filtering methodology.

## View-selection and hardware boundary

I4.3C deliberately introduces **no permanent global view cap**. Every authoritative registered view is available/selected by the adapter contract. Adaptive, explainable, hardware-aware view selection and batching are deferred to I4.3F.

A voxel-capacity safety bound may mark an execution `DEGRADED` if new voxels are dropped; it is not a frame/view-selection policy and does not authorize silent quality reduction. Full RTX 2060/4090/5090 resource adaptation remains later qualification/productization work.

The current I4.3C fusion binding explicitly requires a single shared SfM camera model. Mixed-camera SfM sessions fail clearly rather than being silently approximated; generalized mixed-camera support can be added only with dedicated evidence/tests.

## Progressive-preview compatibility

I4.3C emits the UI-neutral I4.3A progress stages for preparation, view selection, DA3 execution, per-view SfM alignment, fusion, and finalization.

I4.3C does not yet stream provisional geometry revisions into the application viewport. The v1.0 progressive-reconstruction requirement remains locked in I4.3A/I6. Native DA3 evidence and final canonical geometry are kept distinct, and unfinished/provider-native geometry is never promoted as canonical authority.

## Qualification semantics

- Provider family: `multi_view_ai_depth`.
- Implementation: `da3`.
- Provider ID: existing `depth-anything-3-external` identity.
- Qualification state: `candidate`.
- Canonical pose source: `pycolmap_sfm`.
- DA3 predicted pose authority: `false`.
- MVS-vs-DA3 production comparison: pending I4.3G.
- Universal provider winner assumed: `false`.

A rejected subset of DA3/SfM alignments yields `DEGRADED` when enough accepted views remain. Insufficient accepted views fails closed. Fusion capacity drops are also reported as degradation rather than silently hidden.

## Validation policy

The I4.3C test gate extends I4.3B and explicitly includes the legacy DA3 tests so the new canonical adapter cannot break the historical DA3 provider surface.

Expected authoritative-repository counts for this overlay are:

- focused: 249 passed;
- stress: 119 passed;
- complete affected matrix: 368 passed;
- I4.1.9 backward sanity: 61 passed.

The exact count is secondary to all selected tests passing, because repository composition may legitimately evolve.

The real binding smoke is execution-free: it validates the current isolated DA3 runtime/source revision/runtime lock against the already-passing I4.3A architecture without starting a DA3 inference. Synthetic unit/integration/stress execution exercises the new SfM-alignment/fusion adapter path. A later bounded real provider execution may be used as an additional closure gate if suitable preserved authoritative SfM input evidence is available; it must not require rebuilding historical evidence merely for repetition.

## Exit gate

I4.3C may close only when:

- focused, stress, and complete I4.2 + I4.3A/B/C affected tests pass;
- legacy DA3 backward tests remain green;
- the isolated DA3 runtime probe passes on the real machine;
- I4.1.9 backward sanity remains green;
- `git diff --check` is clean;
- the real I4.3A/DA3-runtime binding smoke passes;
- provider qualification remains `candidate`, with comparison/promotion deferred to I4.3G/I4.3H.

On PASS, I4.3D — RollingDepth + Video Depth Anything temporal-provider work — is next.
