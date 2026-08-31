# I4.3G4C — Single-Image Matched-View Manual Review & G4 Closure

## Scope

I4.3G4C consumes only the three frozen I4.3G4B production-native scenario snapshots. It does **not** rerun PyCOLMAP or any of the six single-image providers.

Production-native evidence remains primary. The matched-view diagnostic is secondary and cannot erase a `degraded` or `rejected` G4B outcome. In particular, the frozen Low Texture rejections remain terminal production-quality evidence even if some native visual characteristic is preferred during manual review.

## Frozen scenarios and providers

Scenarios:

- `g4_single_image_complete_room`
- `g4_single_image_low_texture`
- `g4_single_image_weak_overlap`

Providers:

- Metric3D v2
- UniDepthV2
- MoGe-2
- Depth Pro
- Depth Anything V2
- Marigold

The seven G1 manual-review dimensions remain unchanged:

- `global_completeness`
- `local_shape_plausibility`
- `edge_definition`
- `small_detail_preservation`
- `texture_sensitivity`
- `cross_view_stability_after_sfm_alignment`
- `false_geometry_and_floaters`

## Fair six-provider review policy

G4C uses one deterministic A-F blind permutation per frozen G4B scenario. The public review package contains only the assignment commitment until the seven-dimension review is complete.

Two complementary diagnostic surfaces are generated from frozen evidence:

1. **Canonical-geometry panels** — every provider with qualified G4B canonical geometry is rendered from one PyCOLMAP-sparse-landmark-derived visual frame, the same fixed camera/framing, and one equal deterministic point draw count. If G4B produced no qualified canonical geometry for a rejected provider, G4C shows a blinded placeholder instead of reconstructing or synthesizing replacement geometry.
2. **Supplemental native-depth panels** — used only for provider frame artifacts that G4B individually content-addressed. The current G4B success path freezes the worker manifest and canonical outputs but not every per-frame NPZ, so G4C deliberately does not consume those unhashed successful-provider frame files. Rejection evidence does hash the retained per-frame NPZs, allowing those rejected providers to be inspected on the exact common selected keyframes without pretending the evidence provenance is stronger than it is. Each included provider uses one fixed provider-internal display range over the whole scenario, with no per-frame auto-fit. Absolute color values are not a cross-provider metric scale.

The reviewer records, per dimension, zero or more preferred blind labels and any genuinely `not_assessable` labels. Multiple preferred labels are allowed. G4C deliberately creates no forced ordering, weighted score, provider ranking, or universal winner.

## Reveal and closure

Reveal is blocked until all seven dimension records are explicitly confirmed and `reviewer_confirmation=true`. The reveal maps A-F labels back to provider identities while preserving every frozen G4B production status.

Formal G4 closure requires controlled reveals for all three scenarios. Closure records:

- all three scenarios accounted;
- all six provider terminal outcomes accounted per scenario;
- all required manual reviews complete;
- G4 production rejection semantics preserved;
- `i4_3g4_closed=true`;
- `i4_3g5_authorized=true`;
- no provider selection or promotion;
- no ranking/universal winner/global scalar score;
- no Auto or Hybrid policy;
- PyCOLMAP remains camera/scene authority;
- no metric-scale, measurement, certified-metrology, or mesh authority.

The next boundary is I4.3G5 cross-provider synthesis and formal G closure.
