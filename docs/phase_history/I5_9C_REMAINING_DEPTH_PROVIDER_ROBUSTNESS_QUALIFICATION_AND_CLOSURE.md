# I5.9C Remaining Depth-Provider Robustness Qualification and Closure

## Scope

I5.9C is evidence-only. It closes I5.9 by consuming the frozen I5.9A admission, the real I5.9B1 replay, the real I5.9B2 replay, and a focused qualification receipt. It performs no provider inference and does not rerun SfM, bundle adjustment, or MVS.

## Frozen current evidence

I5.9B1 contains 24 terminal provider/scenario attempts for Metric3D v2, UniDepthV2, MoGe-2, and Depth Pro across the six admitted Castle/PyCOLMAP scenarios. The observed B1 matrix is 24 PASS, 0 DEGRADED, 0 REJECTED, with all selected registered views accepted and no fusion-capacity drops.

I5.9B2 contains 12 terminal provider/scenario attempts for Depth Anything V2 and Marigold. The observed B2 matrix is 11 PASS, 1 DEGRADED, 0 REJECTED. The one degraded result is Marigold on `castle_dynamic_proxy`: 19 selected and processed views, 18 accepted views, one alignment-rejected view, and zero fusion-capacity drops. The provider-specific alignment semantics remain `per_view_affine_inverse_depth_to_authoritative_sfm` for Depth Anything V2 and `per_view_affine_native_domain_to_authoritative_sfm` for Marigold.

## Historical evidence remains active

I4.3G4 is a distinct historical evidence axis. Its Low Texture results remain:

- Metric3D v2: PASS
- UniDepthV2: PASS
- MoGe-2: PASS
- Depth Pro: PASS
- Depth Anything V2: REJECTED
- Marigold: REJECTED

The current Castle campaign does not overwrite or upgrade those historical scenario-specific outcomes.

## Qualification statement

The current six-provider Castle campaign is execution-complete with 36/36 terminal attempts: 35 PASS, 1 DEGRADED, and 0 REJECTED. This supports a bounded current frozen-foundation robustness envelope only. It does not establish universal provider robustness, a provider ranking, a universal winner, cross-GPU equivalence, ground-truth dense-depth accuracy, ground-truth surface accuracy, physical metric scale, or certified metrology.

The Marigold dynamic-proxy degradation is preserved as alignment robustness evidence. It is not reclassified as a fusion-capacity limitation and is not retuned or retried.

## Authority and non-goals

PyCOLMAP remains the sole camera/global-frame authority. No native/model intrinsics become canonical authority. No provider is selected or promoted. No global scalar score is created. No threshold, fusion capacity, or production default is changed. No Auto or Hybrid policy is introduced.

## Roadmap transition

I5.9 closes after successful I5.9C formal evidence generation. I5.10 MP-SfM evaluation becomes authorized but remains not started. I5.11 through I5.14 remain deferred.
