# I5.9B1 — Metric-like single-image provider robustness replay

## Scope

I5.9B1 is the first real replay batch authorized by I5.9A. It covers exactly:

- Metric3D v2
- UniDepthV2
- MoGe-2
- Depth Pro

across exactly the six current frozen Castle/PyCOLMAP foundations:

- `castle_control`
- `castle_low_overlap`
- `castle_weak_parallax`
- `castle_overexposure`
- `castle_symmetry`
- `castle_dynamic_proxy`

This yields 24 provider/scenario attempts. Execution is sequential and each attempt is immutable.

## Why these four providers are one batch

All four retain the existing E-series metric-like single-image production contract and the same canonical alignment family:

`per_view_sparse_correspondence_scale_to_authoritative_sfm`

Native metric-like depth is not promoted to physical metric scale. Model-estimated intrinsics are not canonical camera authority. PyCOLMAP remains the sole camera/global-frame authority.

Depth Anything V2 and Marigold are not executed here; they remain I5.9B2 because their existing relative-depth production paths use provider-specific affine alignment.

## Preflight

`preflight` mode is execution-free. It must:

1. bind the exact I5.9A admission SHA;
2. bind the exact I5.6A admission SHA;
3. bind the fresh I5.9B1 validation receipt;
4. re-hash all six frozen scenario SfM report bindings;
5. revalidate registered-image and selected-sparse integrity through the I5.6A analysis contract;
6. probe all four production runtimes and their source/checkpoint identity;
7. resolve the hardware profile through the existing `ComputeProfile.AUTO` path;
8. resolve all 24 I4.3F adaptive plans without starting provider inference;
9. verify the four worker scripts exist and bind their SHA256 values.

Preflight cannot run SfM, MVS, or any provider inference.

## Production-native replay

`production-native` mode repeats the preflight in a fresh output root and then executes exactly one attempt per admitted provider/scenario pair.

For each attempt it must:

- project the frozen SfM evidence to registered camera views only, without modifying camera transforms or the global frame;
- resolve the existing I4.3F adaptive plan before provider execution;
- run the existing E1/E2/E3/E4 production adapter and worker;
- preserve the provider's existing terminal status semantics;
- persist native worker evidence, alignment evidence, canonical geometry when produced, adaptive-plan identity, runtime configuration, wall time, and device-wide peak VRAM observation;
- re-hash the authoritative source/SfM inputs after execution;
- record fusion-capacity drops as evidence only.

### Terminal scientific outcomes

The valid terminal outcomes are:

- `pass`
- `degraded`
- recognized existing quality `rejected`

A recognized quality rejection is preserved and the campaign continues to the next admitted provider. It is not retried or tuned away.

A runtime/integrity exception is not a scientific terminal outcome. It writes failure evidence, stops the batch nonzero, and remains immutable.

## Fusion-capacity rule

The existing production fusion capacity remains 1,500,000 voxels.

`dropped_new_voxels_capacity` is recorded when available, but **does not automatically change a single-image provider's terminal status**. This intentionally differs from the I5.8 Video Depth Anything robustness contract and preserves the E1–E4 production semantics admitted by I5.9A.

## Frozen safety policy

I5.9B1 does not authorize:

- provider comparison or ranking;
- a global scalar score or universal winner;
- result-driven retuning;
- retry-until-pass;
- threshold relaxation;
- production-default modification;
- fusion-capacity increase;
- a permanent global view cap;
- SfM or bundle-adjustment reruns;
- MVS reruns;
- MP-SfM;
- Auto provider selection;
- hybrid reconstruction;
- physical metric-scale or certified-metrology claims;
- surface-mesh generation.

## Next

After I5.9B1 is executed, validated, frozen, committed and pushed, I5.9B2 may execute the two relative-depth providers. I5.9C remains deferred until both B1 and B2 are complete.
