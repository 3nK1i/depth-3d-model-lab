# I5.8C — Video Depth Anything Robustness Qualification and Closure

## Purpose

I5.8C is the evidence-only closure of the Video Depth Anything robustness workstream. It consumes the frozen I5.8A admission and the exact I5.8B real replay. It performs no Video Depth Anything inference and does not modify any production provider setting.

## Frozen predecessor evidence

- I5.8A admission SHA256: `1d635cdcefe02e1ee3b66dfaa8af4df49fb952e58cbf4446b50c2d562369d90a`
- I5.8B real replay SHA256: `24eab2c95869ea96792c6599eeb94ce2801d5920d4a37b9a015a5428e568bb63`
- Resolved VDA checkpoint SHA256: `3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33`
- VDA source revision: `4f5ae23172ba60fd7bc11ef671cca678842c7072`

## Current real-replay result

The I5.8B campaign completed all six admitted Castle temporal scenarios with no runtime/infrastructure failure. Every registered frame was processed and accepted by the per-frame PyCOLMAP-SfM alignment path. Nevertheless, every scenario persisted `DEGRADED` quality because the existing production fusion capacity was reached.

The frozen production maximum is 1,500,000 retained voxels. The I5.8B observations are:

| Scenario | Alignment | Fusion capacity drops | Quality |
|---|---:|---:|---|
| castle_control | 19/19 accepted | 3,678,651 | DEGRADED |
| castle_low_overlap | 6/6 accepted | 215,631 | DEGRADED |
| castle_weak_parallax | 7/7 accepted | 925,581 | DEGRADED |
| castle_overexposure | 19/19 accepted | 3,518,689 | DEGRADED |
| castle_symmetry | 12/12 accepted | 3,213,160 | DEGRADED |
| castle_dynamic_proxy | 19/19 accepted | 3,744,324 | DEGRADED |

I5.8C does not increase that capacity, relax any threshold, rerun any scenario, or reinterpret these results as clean quality PASS.

## Historical temporal evidence preserved

The frozen G3 Video Depth Anything anchors remain active:

- static sweep: PASS
- motion: DEGRADED
- low texture: PASS

The historical motion degradation is not overwritten by the current Castle replay.

## Qualification statement

I5.8C closes I5.8 as an **execution-qualified but quality-DEGRADED, capacity-bounded operating envelope** for the admitted current frozen-foundation scenarios on the observed RTX 2060 `efficient / ViT-S / fp16` runtime profile.

This is not a universal robustness claim. In particular, I5.8C does not claim:

- clean VDA quality PASS on the current six-scenario replay;
- universal Video Depth Anything robustness;
- cross-GPU equivalence or RTX 4090/5090 qualification;
- long-sequence resource robustness;
- provider-specific cancellation qualification;
- dense-depth ground-truth accuracy or surface-accuracy qualification;
- physical metric scale or metrology authority;
- provider ranking, promotion, or automatic selection.

## Authority locks

- PyCOLMAP remains the only camera-pose/global-frame authority.
- VDA estimated intrinsics are not canonical camera authority.
- Native metric-depth output is not promoted to physical metric scale.
- No provider ranking/promotion/selection is introduced.
- No permanent global view cap is introduced.
- No SfM, bundle-adjustment, or MVS rerun occurs.
- No fusion-capacity increase, threshold change, production-default mutation, or retry-until-pass occurs.

## Roadmap decision

If the exact frozen I5.8A/I5.8B evidence and the I5.8C regression receipt validate, I5.8 is closed and I5.9 Remaining depth-provider robustness is authorized but not started.
