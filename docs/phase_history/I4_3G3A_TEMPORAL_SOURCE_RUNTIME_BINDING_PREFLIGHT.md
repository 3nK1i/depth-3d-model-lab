# I4.3G3A — Temporal Source / Runtime Binding Preflight

## Scope

I4.3G3A is the smallest execution-free opening boundary for temporal-provider qualification. It binds the three frozen G1 temporal scenarios to explicit, content-addressed recorded-video sources and verifies the existing Video Depth Anything and RollingDepth runtime readiness without starting SfM or either dense backend.

The required scenarios remain exactly:

- `g3_temporal_static_sweep`
- `g3_temporal_motion`
- `g3_temporal_low_texture`

No scenario semantics are inferred from a path or filename. Supplying a source through the scenario-specific preflight argument is the explicit engineering binding. Each source is persisted with a full-file SHA256; the production reader fingerprint, container-reported frame count and frame rate are retained as supporting provenance.

## Reused architecture

G3A does not add another temporal provider architecture. It reuses the existing I4.3D1 Video Depth Anything provider, I4.3D2 RollingDepth provider, I4.3D3 temporal closure, and I4.3F post-SfM adaptive execution projection.

Production-native G3 evidence remains the primary operating-envelope comparison. Each provider keeps its provider-specific I4.3F plan while preserving source order. A production-native plan is not required to be contiguous.

Matched-view evidence remains a secondary causal diagnostic. For temporal providers it must use one common **contiguous source-ordered subsequence**, preserve each provider's native temporal execution semantics, use the same authoritative SfM identity, use the same review camera/framing, and forbid independent provider auto-fit.

## Authority and execution locks

I4.3G3A starts no PyCOLMAP SfM, VDA, or RollingDepth execution and consumes no provider quality result. PyCOLMAP remains the only camera/global-frame authority. Providers remain candidates. Auto is not implemented; hybrid is not authorized; no metric-scale, measurement, certified-metrology, or surface-mesh authority is introduced; no permanent global view cap is added.

## Closure boundary

A ready G3A snapshot authorizes only the next engineering boundary: **I4.3G3B production-native temporal execution**, one frozen scenario at a time. Matched-view diagnostics and manual temporal A/B review remain deferred until production-native evidence exists.
