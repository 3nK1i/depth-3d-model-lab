# I5.8A - Video Depth Anything Robustness Admission + Envelope Design

## Scope

I5.8A is execution-free. It consumes the formally closed I5.7 artifact, the exact six-case I5.6A Castle/PyCOLMAP admission matrix, and the frozen I4.3G3C temporal-provider closure to define the real Video Depth Anything replay authorized for I5.8B.

I5.8A does not execute Video Depth Anything, RollingDepth, PyCOLMAP SfM, COLMAP MVS, or another dense provider. It does not tune thresholds or production defaults, select/rank providers, create Auto or Hybrid, pull MP-SfM forward, or create metric/measurement/metrology authority.

## Frozen predecessor identities

- I5.7C closure SHA256: `b76beeb82db0a4477fdea9212ac3f693725d9b57c8c3fafdf307eae4438a50fc`
- I5.6A Castle/SfM admission SHA256: `506fb778d82ddc46350cbe4922ebe8a2ba43e0df4fa7818687a5768ebe0afc7a`
- I4.3G3C temporal closure SHA256: `18f1c54d421277cf4dd1cf81a96e0bb74b5b1846a9d38f099fdd0222f5f350bd`

The historical VDA G3 production statuses remain immutable evidence:

- `g3_temporal_static_sweep`: `pass`
- `g3_temporal_motion`: `degraded`
- `g3_temporal_low_texture`: `pass`

I5.8A does not authorize rerunning those historical G3 scenarios. The historical motion degradation remains active evidence even if a new Castle dynamic-proxy replay later passes.

## VDA-specific production contract

I5.8 deliberately does not reuse RollingDepth's sequence-global alignment semantics.

The existing production VDA path is preserved exactly:

- provider id `video-depth-anything-temporal`;
- provider family `temporal_depth`;
- external-process execution;
- official upstream streaming-cache temporal mode;
- one model load per worker sequence;
- warm-up followed by an explicit sequence-state reset, so warm-up is never the first user frame;
- ordered registered SfM views with one fixed worker session and monotonically increasing worker frame ids;
- one stable pixel grid / shared SfM camera model within a sequence;
- native VDA depth and model-estimated intrinsics preserved as model evidence;
- each registered VDA depth map aligned independently into the authoritative PyCOLMAP SfM gauge;
- no RollingDepth-style sequence-global affine is introduced;
- VDA native metric-depth semantics do not establish physical metric scale in canonical reconstruction;
- VDA estimated intrinsics are not canonical projection authority;
- canonical projection uses bundle-adjusted PyCOLMAP poses;
- permanent global view cap remains `None`.

The frozen production alignment/fusion configuration remains:

- minimum correspondences: `8`;
- maximum relative residual: `0.35`;
- voxel scene fraction: `0.004`;
- pixel stride: `2`;
- maximum voxels: `1,500,000`;
- minimum accepted views: `2`.

## Hardware-resolved model variants

VDA is not frozen to one universal checkpoint across all supported GPUs. I5.8A preserves the existing production profile resolver:

- efficient: ViT-S, metric VDA Small, input size 392, max resolution 960;
- balanced: ViT-B, metric VDA Base, input size 448, max resolution 1280;
- ultra: ViT-L, metric VDA Large, input size 518, max resolution 1920.

I5.8B must record the current hardware-resolved production profile and verify that the probed checkpoint is exactly the checkpoint selected by that profile. It may not silently switch model size/checkpoint to obtain a better result.

## I5.8B admitted real replay matrix

The same six frozen authoritative Castle/PyCOLMAP scopes are scientifically applicable to the production VDA registered-keyframe stream, but their interpretations are VDA-specific:

1. `castle_control` - streaming-cache registered-sequence control.
2. `castle_low_overlap` - registered-keyframe spacing/cache-continuity stress; this is not claimed to be an MVS overlap metric and not claimed to reproduce continuous full-video timing.
3. `castle_weak_parallax` - narrow-baseline streaming temporal-stability stress.
4. `castle_overexposure` - photometric overexposure streaming temporal stress.
5. `castle_symmetry` - repetitive-structure streaming stress on the frozen authoritative registered subset only (12/19).
6. `castle_dynamic_proxy` - dynamic/motion-boundary streaming-cache stress.

The production reconstruction provider receives only the authoritative registered keyframes, in source order. I5.8A therefore explicitly does not claim that these registered-keyframe sequences equal uninterrupted original-video temporal continuity.

## Explicit exclusions/deferments

- `castle_blur`: blocked upstream because no reusable authoritative SfM exists; not a VDA failure.
- `castle_underexposure`: blocked upstream; not a VDA failure.
- `castle_clutter`: not formally admitted because the frozen predecessor did not revalidate the intended clutter semantics.
- `kitti_forward_motion`: blocked upstream because current authoritative PyCOLMAP produced no registered sparse model; MP-SfM remains I5.10.
- difficult-handheld Horse: scientifically relevant but not admitted because the exact temporal source plus authoritative SfM binding is not present in the I5.6A matrix.

## I5.8B evidence contract

Every real attempt must use a new timestamped immutable evidence directory. Required evidence includes runtime/profile/checkpoint identity, source and SfM integrity, ordered registered-view identity, stable-grid contract, native VDA depth stack, native per-frame latency records, per-registered-frame alignment records, and canonical fusion evidence when minimum alignment support is met.

PASS and DEGRADED map to the existing provider terminal result. REJECTED is allowed only for an explicit scientific provider-quality failure after runtime/integrity checks succeed; runtime, environment, source-integrity, authority, or orchestration failures are not scientific rejections and fail the attempt non-zero.

No retry-until-pass and no result-driven retuning are authorized. Missing performance/resource observations remain missing rather than being reconstructed or invented.

## Authority

PyCOLMAP remains the sole camera/global-frame authority. VDA receives no physical metric-scale authority, measurement authority, certified-metrology authority, mesh authority, provider selection, or provider promotion. Its native metric-depth and estimated-intrinsics outputs remain provider evidence only.

## Next boundary

A passing I5.8A admission artifact authorizes only **I5.8B - real frozen-foundation Video Depth Anything robustness replay**. I5.9, I5.10 MP-SfM, I5.11 recovery hardening, I5.12 Hybrid, I5.13 Auto, and I5.14 final I5 closure remain deferred to their locked roadmap positions.
