# I5.7A - RollingDepth Robustness Admission + Envelope Design

## Scope

I5.7A is execution-free. It binds the frozen I5.6 closure, the exact six-case I5.6A Castle/PyCOLMAP admission matrix, and the formally closed I4.3G3C temporal-provider evidence to define the real RollingDepth replay authorized for I5.7B.

I5.7A does not execute RollingDepth, PyCOLMAP SfM, COLMAP MVS, DA3, or Video Depth Anything. It does not tune thresholds or production defaults, select/rank providers, create Auto or Hybrid, pull MP-SfM forward, or create metric/measurement/metrology authority.

## Frozen predecessor identities

- I5.6A admission SHA256: `506fb778d82ddc46350cbe4922ebe8a2ba43e0df4fa7818687a5768ebe0afc7a`
- I5.6C closure SHA256: `448414d95e4e9c87b77ca5b3a5b94f6b2b22bb595d1ec81df8f2ec48c1424ccb`
- I4.3G3C closure SHA256: `18f1c54d421277cf4dd1cf81a96e0bb74b5b1846a9d38f099fdd0222f5f350bd`

The G3C RollingDepth production statuses remain immutable historical anchors:

- `g3_temporal_static_sweep`: `degraded`
- `g3_temporal_motion`: `rejected`
- `g3_temporal_low_texture`: `pass`

I5.7A does not authorize rerunning those historical G3 scenarios.

## I5.7B admitted real replay matrix

The following exact authoritative PyCOLMAP scopes are admitted for a new RollingDepth-only replay:

1. `castle_control` - temporal-sequence control.
2. `castle_low_overlap` - temporal-support/spacing stress. This is not claimed to be equivalent to an MVS overlap metric.
3. `castle_weak_parallax` - narrow-baseline temporal geometry stress.
4. `castle_overexposure` - photometric overexposure temporal stress.
5. `castle_symmetry` - registered authoritative subset only (12/19 in the frozen predecessor).
6. `castle_dynamic_proxy` - dynamic/motion-boundary temporal stress.

The current RollingDepth production path remains stateful and source-order dependent. It uses one sequence-global affine alignment to the authoritative PyCOLMAP gauge and forbids per-frame fitting.

## Explicit exclusions/deferments

- `castle_blur`: blocked upstream; no authoritative reusable SfM. Not a RollingDepth failure.
- `castle_underexposure`: blocked upstream; no authoritative reusable SfM. Not a RollingDepth failure.
- `castle_clutter`: not formally admitted because the predecessor did not revalidate the intended clutter stress semantics.
- `kitti_forward_motion`: blocked upstream because the current authoritative PyCOLMAP path produced no sparse model/registered cameras. MP-SfM remains deferred to I5.10.
- difficult-handheld Horse: scientifically relevant but not admitted in I5.7A because the exact temporal source + authoritative SfM binding is not present in the I5.6A admission matrix. I5.7A does not invent that provenance.

## Frozen RollingDepth production defaults

I5.7A preserves the current provider defaults, including:

- provider id `rolling-depth-temporal`;
- default device `cuda:0`;
- default precision `fp16`;
- processing resolution `512`;
- dilations `[1, 25]`;
- snippet lengths `[3]`;
- max VAE batch size `1`;
- snippet unloading enabled;
- permanent global view cap `None`.

I5.7B must verify the pinned external runtime/source/checkpoint before any real execution. Missing performance/resource evidence remains missing rather than being fabricated.

## I5.7B evidence contract

Every real attempt must use a new timestamped evidence directory and preserve failed, interrupted, blocked, degraded, rejected and passing attempts. Required evidence includes source/SfM integrity, temporal ordering identity, provider-native output, global sequence-alignment evidence, canonical fusion evidence when alignment is accepted, wall-time observation, and resource observations when available.

PASS, DEGRADED and REJECTED are all legitimate terminal scientific outcomes. No retry-until-pass and no result-driven retuning are authorized.

## Authority

PyCOLMAP remains the sole camera/scene-frame authority. RollingDepth receives no pose authority, physical metric-scale authority, measurement authority, certified-metrology authority, or mesh authority. Canonical fused geometry remains spatially subordinate to the authoritative PyCOLMAP cameras and arbitrary reconstruction gauge.

## Next boundary

A passing I5.7A admission artifact authorizes only **I5.7B - real frozen-foundation RollingDepth robustness replay**. I5.8 Video Depth Anything, I5.10 MP-SfM, I5.11 OOM/disk/crash/recovery, I5.12 Hybrid, and I5.13 Auto remain deferred to their locked roadmap positions.
