# I4.3D3 — Temporal Provider Architecture Closure

## Scope

I4.3D3 is closure-only. It validates the passing I4.3D1 Video Depth Anything and I4.3D2 RollingDepth binding evidence and freezes the common `temporal_depth` provider contract without selecting or promoting either implementation.

## Frozen architecture

- PyCOLMAP SfM remains the sole authoritative camera/global reconstruction frame.
- Video Depth Anything and RollingDepth remain `candidate` temporal-depth providers.
- Both produce depth evidence that is aligned/projected into the authoritative SfM frame.
- Video Depth Anything retains official streaming-cache semantics; model-estimated intrinsics and model metric-depth semantics are not promoted to canonical authority.
- RollingDepth retains multi-dilation snippet co-alignment and one sequence-level affine alignment to the SfM reconstruction gauge; no per-frame alignment is allowed and the affine fit does not establish physical metric scale.
- No permanent global dense-view cap is introduced. Adaptive, explainable frame/view selection remains I4.3F.
- Progressive status remains required by the canonical I4.3A contract.

## Authority limits

I4.3D3 does not start a backend, select or promote a provider, implement Auto, authorize a hybrid, establish metric scale, create measurement/certified-metrology authority, or generate a surface mesh.

## Qualification boundary

No temporal provider is declared universally superior. Temporal video A/B and operating-envelope qualification remain required in I4.3G/I5. I4.3D3 only closes the temporal-provider architecture surface and authorizes I4.3E.

## Next

I4.3E adapts the remaining installed depth-model providers individually. I4.3F adds adaptive frame/view selection. I4.3G performs controlled cross-provider qualification, including temporal video comparison. I4.3H closes the full provider architecture.
