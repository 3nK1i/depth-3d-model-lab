# I4.3A — Canonical Dense-Provider Architecture Foundation v2

## Scope

I4.3A is an execution-free architecture step after the successful I4.2F TempleRing closure. It freezes the provider vocabulary, authority rules, progressive-preview contract, and I4.3B-H integration/qualification roadmap before any production provider adapter is bound.

The v2 correction deliberately separates three concepts that must not be conflated:

1. **Selection mode**
   - `Auto` — policy only; never an executable backend identity.
   - `Explicit Provider` — an engineer/user explicitly requests one concrete provider implementation.
2. **Provider family / methodology**
   - classical MVS;
   - multi-view AI depth;
   - temporal depth;
   - single-image depth;
   - future evidence-gated hybrid.
3. **Concrete provider implementation**
   - COLMAP MVS;
   - Depth Anything 3 (DA3);
   - RollingDepth;
   - Video Depth Anything;
   - Metric3D v2;
   - UniDepthV2;
   - MoGe-2;
   - Depth Pro;
   - Depth Anything V2;
   - Marigold.

No backend is probed or started, no provider is selected/promoted, no Auto policy is implemented, and no application/UI binding is performed in I4.3A.

## Repository inspection basis

The foundation was designed around the existing reconstruction stack rather than introducing a second pipeline:

- `core/reconstruction/orchestration.py` already owns backend-neutral stage/provider concepts.
- `core/reconstruction/mvs_execution.py` and `core/reconstruction/mvs_camera_bridge.py` contain the qualified MVS execution and authoritative-camera work that I4.3B will adapt.
- `core/reconstruction/da3_provider.py` and `core/reconstruction/da3.py` contain the existing isolated DA3 execution surface that I4.3C will adapt **as dense/depth evidence only**; PyCOLMAP remains pose authority.
- Existing application depth adapters include the target surfaces such as `core/models/adapters/video_depth_anything_adapter.py` and the individually qualified single-image model adapters in the complete repository. The compact handoff intentionally omits parts of historical `core.models`, so I4.3A does not import those adapters.
- `core/reconstruction/dense_scene_persistence.py` remains the canonical dense persistence destination.
- `app/reconstruction_controller.py` and `app/reconstruction_dense_provider_presentation.py` remain unchanged until the later product-integration step.

RollingDepth is a roadmap provider identity in I4.3A; its adapter is **not fabricated as already bound**. Availability and qualification are runtime/evidence facts in later steps.

## Locked provider-family semantics

### MVS

`COLMAP_MVS` belongs to `mvs` and produces dense geometry directly in the authoritative PyCOLMAP SfM frame.

### Multi-view AI depth

`DA3` belongs to `multi_view_ai_depth`. Even though the historical DA3 provider can emit pose-like evidence, I4.3 production architecture does **not** allow DA3 to replace PyCOLMAP camera/global-frame authority. DA3 contributes depth/supporting geometry that must be aligned/fused into the authoritative SfM frame.

### Temporal depth

`RollingDepth` and `VideoDepthAnything` belong to `temporal_depth`. Their temporal stability/flicker/drift properties are qualified separately from single-frame models.

### Single-image depth

`Metric3D_v2`, `UniDepthV2`, `MoGe_2`, `DepthPro`, `DepthAnythingV2`, and `Marigold` belong to `single_image_depth`. Strong monocular depth performance does not automatically imply strong multi-view reconstruction performance; each must be qualified individually.

### Hybrid

A `hybrid` family is reserved in the contract only as a future hypothesis. No concrete hybrid provider is registered or authorized by I4.3A. A later hybrid experiment is allowed only if cross-provider evidence shows that MVS is cleaner while AI depth meaningfully recovers legitimate MVS holes.

## Progressive/live reconstruction contract

The v1.0 requirement that reconstruction be visible **while processing** is now architectural rather than deferred UI polish.

I4.3A defines a UI-neutral `DenseProviderProgressEvent` contract containing:

- generic stage plus provider-specific stage detail;
- normalized progress fraction;
- processed/total view counts;
- optional resource telemetry (VRAM/RAM);
- explicit degradation/fallback reason fields;
- optional provisional geometry reference;
- monotonic preview revision identity.

Critical authority rule:

> Progressive preview geometry is replaceable, provisional, and never authoritative. Final canonical geometry remains a separate completed-provider result in the PyCOLMAP SfM frame.

This allows later I6 UI work to show MVS depth-map/fusion progress, DA3/AI alignment progress, and progressively available geometry without pretending unfinished evidence is final.

I4.3A does **not** implement the viewport, controller bridge, worker event transport, or provisional fusion policy. It freezes the contract they must use later.

## Locked architecture rules

1. PyCOLMAP SfM remains the sole authoritative camera/global reconstruction frame.
2. MVS operates directly in that authoritative frame.
3. Learned depth remains supporting depth evidence until aligned/fused into that frame.
4. Dense-provider execution cannot itself establish physical metric scale, measurement authority, certified metrology, or surface-mesh authority.
5. `Auto` is a selection policy, never a persisted executable provider ID.
6. Explicit selection targets a concrete provider implementation, not a vague family.
7. Provider failure and degradation are explicit, distinct states.
8. Fallback must be explicit in the selection decision/provenance.
9. Production view selection must be adaptive and engineer-visible.
10. Historical fixed `40`-view experimental limits must never become a permanent global production cap.
11. Engineer override is represented by contract; policy/UX behavior is later work.
12. Provider qualification/promotion is scenario-specific and remains separate from adapter binding.
13. No universal provider winner is assumed.
14. I4.2's `d12_sparse` result is preserved; no production filter change is introduced here.
15. Progressive status is required by the provider-family contract; provisional geometry is never final authority.

## I4.3B-H roadmap frozen by I4.3A

- **I4.3B — MVS production provider adapter**
  Bind the already-qualified COLMAP/PyCOLMAP MVS path behind the canonical contract.
- **I4.3C — DA3 production provider adapter**
  Bind DA3 as multi-view AI dense/depth evidence while preserving PyCOLMAP pose authority.
- **I4.3D — Temporal depth providers**
  RollingDepth + Video Depth Anything.
- **I4.3E — Remaining installed depth-provider adapters/qualification**
  Metric3D v2, UniDepthV2, MoGe-2, Depth Pro, Depth Anything V2, Marigold.
- **I4.3F — Adaptive frame/view selection**
  Replace experimental fixed frame/view caps with explainable, resource-aware selection.
- **I4.3G — Controlled cross-provider scenario qualification**
  Includes the required complete-room **MVS vs DA3** comparison, TempleRing/House evidence, low-texture and weak-overlap cases, and temporal video comparison. Conclusions remain scenario-specific.
- **I4.3H — Provider architecture closure**
  Record which providers are integrated, qualified, experimental, their operating envelopes/failure modes/resource characteristics, and freeze I4.3.

The first real Auto-selection policy belongs to I5 after robustness evidence exists; I4.3 does not hard-code a winner.

## Not performed

I4.3A does **not**:

- start COLMAP/PyCOLMAP, DA3, RollingDepth, Video Depth Anything, or any installed depth model;
- run PatchMatch, Stereo Fusion, SfM, BA, alignment, fusion, or filtering;
- bind a real provider adapter;
- implement Auto selection or hardware adaptation;
- implement progressive UI rendering;
- change persistence/controller/UI behavior;
- change production filtering;
- promote a provider or hybrid method;
- claim metric scale, measurement authority, certified metrology, or mesh generation.

## Next

**I4.3B — MVS Production Provider Adapter** is the next narrow implementation step after I4.3A validation.
