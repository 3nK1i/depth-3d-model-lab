# I5.9A - Remaining Depth-Provider Robustness Inventory and Admission

## Status

I5.9A is an execution-free inventory, predecessor-binding, and robustness-admission step. It does not execute any single-image depth provider, rerun SfM/MVS, compare providers, tune thresholds, increase fusion capacity, or modify production defaults.

## Frozen predecessor

I5.9A starts only after formal I5.8 closure. The exact I5.8C closure artifact is frozen by SHA256:

- I5.8C SHA256: `741ecf17db05572b86102b410c219559edaacd7a64bf7ea4108eac8bda57daa2`
- I5.8 conclusion: `CLOSED / PASS`, with `quality_status=degraded` and the VDA capacity-bounded limitation preserved.
- I5.9 is authorized and not previously started.

I5.9A additionally binds:

- I5.6A Castle/PyCOLMAP scenario admission SHA256: `506fb778d82ddc46350cbe4922ebe8a2ba43e0df4fa7818687a5768ebe0afc7a`
- I4.3G4C single-image provider qualification closure SHA256: `577dba2b6f4549e122ac1d6892d537d7ecac49049410be819fbf1349f7d4abd2`

## Remaining provider inventory

Repository inspection identifies exactly six remaining production single-image depth providers:

1. `metric3d_v2` - `metric3d-v2-single-image`
2. `unidepth_v2` - `unidepth-v2-single-image`
3. `moge_2` - `moge-2-single-image`
4. `depth_pro` - `depth-pro-single-image`
5. `depth_anything_v2` - `depth-anything-v2-single-image`
6. `marigold` - `marigold-single-image`

All six production adapters are stateless single-image providers. Existing production configuration remains frozen:

- minimum sparse correspondences: `8`
- maximum relative residual: `0.35`
- voxel scene fraction: `0.004`
- pixel stride: `2`
- maximum retained voxels: `1,500,000`
- minimum accepted aligned views: `2`
- all registered views available by default: `true`
- adaptive selection deferred to existing `I4.3F`
- permanent global view cap: `None`

Fusion-capacity drops are preserved as evidence but do **not** change the existing single-image provider terminal-status semantics. I5.9A does not mechanically copy the VDA I4.3D1 capacity-degradation rule into the I4.3E1–E6 adapters.

## Scientifically justified real-replay subdivision

I5.9B is split into two provider batches because the production alignment contracts are materially different.

### I5.9B1 - metric-like native depth / sparse scale alignment

- `metric3d_v2`
- `unidepth_v2`
- `moge_2`
- `depth_pro`

Existing canonical alignment family:

`per_view_sparse_correspondence_scale_to_authoritative_sfm`

Native metric-like depth remains evidence only. It does not establish physical metric scale in the arbitrary PyCOLMAP reconstruction gauge, and model-estimated intrinsics never replace the authoritative PyCOLMAP cameras.

### I5.9B2 - relative depth / provider-specific affine alignment

- `depth_anything_v2`
- `marigold`

Existing canonical alignment semantics:

- Depth Anything V2: `per_view_affine_inverse_depth_to_authoritative_sfm`
- Marigold: `per_view_affine_native_domain_to_authoritative_sfm`

These relative-depth outputs are never promoted to physical metric scale.

The historical I4.3G4C evidence independently supports the subdivision: all four B1 providers passed the Low Texture production anchor, while Depth Anything V2 and Marigold were the two preserved Low Texture production rejections. No historical rejection is overwritten by I5.9A.

## Current admitted robustness matrix

The exact six current Castle/PyCOLMAP foundations are reused without SfM rerun:

| Scenario | Registered / input | I5.9 single-image interpretation |
|---|---:|---|
| `castle_control` | 19 / 19 | stateless registered-view single-image dense control |
| `castle_low_overlap` | 6 / 7 | reduced registered-view support, alignment and fusion robustness; not MVS-overlap equivalence |
| `castle_weak_parallax` | 7 / 7 | narrow-baseline sparse-SfM alignment-support robustness |
| `castle_overexposure` | 19 / 19 | photometric overexposure single-frame depth robustness |
| `castle_symmetry` | 12 / 19 | repetitive structure plus sparse-support alignment robustness |
| `castle_dynamic_proxy` | 19 / 19 | independent-frame dynamic contamination robustness; no temporal-stability claim |

Each admitted scenario applies to both B1 and B2, while preserving provider-specific alignment/runtime semantics.

## Excluded or deferred scenarios

- `castle_blur`: `BLOCKED_UPSTREAM`; no authoritative PyCOLMAP SfM foundation after blur stress.
- `castle_underexposure`: `BLOCKED_UPSTREAM`; pre-SfM ingestion rejection prevents an authoritative SfM foundation.
- `castle_clutter`: `NOT_FORMALLY_ADMITTED`; clutter stress is not present in the frozen I5.6A matrix.
- `kitti_forward_motion`: `BLOCKED_UPSTREAM`; no authoritative sparse model/registered cameras; MP-SfM remains deferred to I5.10.
- `difficult_handheld_horse`: `DEFERRED_NOT_ADMITTED`; the exact authoritative SfM binding is not in I5.6A and provenance is not invented.

None of these classifications is a remaining-provider failure.

## Historical I4.3G4C anchors

The historical three-scenario production-native results remain authoritative context:

- Complete Room: all six providers `PASS`.
- Low Texture: Metric3D v2, UniDepthV2, MoGe-2 and Depth Pro `PASS`; Depth Anything V2 and Marigold `REJECTED`.
- Weak Overlap: all six providers `PASS`.

Matched-view diagnostics remain secondary; production-native evidence remains primary. There is no provider selection, promotion, ranking, global scalar score, or universal winner.

## I5.9B evidence contract

The real replay must preserve the following:

- fresh immutable attempt directory per provider/scenario;
- failures, interruptions, blocks and rejections remain preserved;
- batch-specific runtime preflight before real execution;
- existing hardware-resolved provider variant/profile and exact source/checkpoint identity recorded;
- authoritative source and PyCOLMAP SfM integrity revalidated;
- existing I4.3F adaptive plan resolved before the result, never retuned from the result;
- per-selected-view provider-native evidence and provider-specific alignment evidence;
- canonical fusion evidence when minimum accepted-view support is met;
- existing single-image terminal semantics preserved (`PASS`, `DEGRADED`, or explicit quality rejection as applicable);
- fusion-capacity drops recorded but not redefined into a new single-image degradation rule;
- runtime/integrity failures remain execution failures rather than being relabeled as scientific rejection;
- no retry-until-pass, no threshold relaxation, no provider ranking, and no result-driven retuning.

## Authority and scientific limits

PyCOLMAP remains the sole authoritative camera and global scene-frame owner. I5.9A does not claim:

- physical metric scale;
- certified metrology;
- authoritative measurement;
- universal single-image provider robustness;
- provider ranking or winner;
- RTX 4090/5090 or cross-GPU equivalence;
- a complete runtime/resource envelope;
- ground-truth dense depth or surface accuracy.

## Roadmap decision

A valid I5.9A artifact marks:

- I5.9A complete;
- I5.9B1 authorized and not started;
- I5.9B2 authorized but executed only after B1 is frozen;
- I5.9C not started;
- I5.10 and later roadmap steps not started.

Next: **I5.9B1 - metric-like single-image provider robustness replay**.
