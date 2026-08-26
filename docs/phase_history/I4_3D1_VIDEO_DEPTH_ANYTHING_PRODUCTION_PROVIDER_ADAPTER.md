# I4.3D1 — Video Depth Anything production provider adapter

## Scope

I4.3D1 is the first half of I4.3D. It binds the already-installed stateful Video Depth Anything runtime behind the I4.3A `temporal_depth` provider contract. RollingDepth is intentionally left for I4.3D2 because it requires a separate official checkout/runtime and a different whole-sequence execution model.

The adapter consumes every registered authoritative SfM view in source temporal order, runs the existing VDA streaming model in an isolated process, preserves the model's native metric-depth / estimated-intrinsics semantics as evidence only, aligns each depth map into the authoritative PyCOLMAP SfM gauge from sparse correspondences, and fuses geometry using only the bundle-adjusted PyCOLMAP poses.

## Authority locks

- PyCOLMAP SfM remains the sole camera/global-frame authority.
- VDA's estimated intrinsics are not used for canonical projection.
- VDA's native metric-depth claim is not promoted to physical metric scale in the canonical reconstruction.
- No measurement, certified metrology, or mesh authority is created.
- No provider promotion or Auto policy is performed.
- All registered views remain eligible; no permanent view cap is introduced.
- Adaptive view selection remains I4.3F.
- Temporal quality/flicker/drift/dynamics qualification remains I4.3G/I5.

## Progressive execution

The adapter emits the I4.3A UI-neutral stages: preparing, view selection, provider execution, alignment, fusion/filtering, and finalization. I4.3D1 does not wire those events into the product UI; that remains part of later reconstruction productization.

## Closure boundary

I4.3D is not complete after I4.3D1. The next substep is I4.3D2: prepare/probe the official RollingDepth isolated runtime and bind it to the same canonical temporal-depth contract without forcing it into VDA's streaming execution model.
