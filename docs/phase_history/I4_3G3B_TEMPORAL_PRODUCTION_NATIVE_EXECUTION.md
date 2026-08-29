# I4.3G3B — Temporal Production-Native Execution

I4.3G3B consumes a hash-verified, ready I4.3G3A source/runtime-binding snapshot and produces new real production-native qualification evidence for the three frozen temporal scenarios using the existing RollingDepth and Video Depth Anything production providers.

## Boundary

- one authoritative PyCOLMAP SfM is created per scenario and shared by both temporal-provider runs;
- each provider receives its own already-qualified I4.3F adaptive production plan derived from that SfM;
- temporal source order and provider-specific stateful semantics are preserved;
- RollingDepth and VDA execute sequentially, never as competing global camera/scene authorities;
- content-addressed source, SfM, adaptive-plan, provider-provenance, artifact and runtime-resource evidence is persisted;
- required G1 metrics are recorded and every unavailable optional metric carries an explicit reason;
- matched-view contiguous-subsequence diagnostics, same-framing manual A/B review and provider verdicts remain I4.3G3C work.

## Authority

PyCOLMAP remains the sole authoritative global camera-pose and scene-frame owner. G3B does not select or promote a provider, implement Auto, authorize Hybrid, claim metric scale, measurement authority, certified metrology, or generate an authoritative surface mesh.

## Real execution workflow

Run `scripts/run_i4_3g3b_temporal_scenario.py` first in `preflight` mode against the exact G3A snapshot and SHA256. Production-native mode is authorized only when the bound source identity, PyCOLMAP runtime, VDA runtime and RollingDepth runtime all remain ready and unchanged. Execute one frozen scenario at a time and preserve the raw output/evidence directory for G3C.

## Terminal provider-quality rejection

A provider quality gate may reject its own production-native output without invalidating the scenario orchestration. In that case G3B records the provider as a terminal `rejected` outcome, preserves the native artifacts and alignment evidence, does not claim canonical geometry for that provider, and continues the other temporal provider against the exact same authoritative PyCOLMAP SfM. Runtime/integrity/authority failures still fail the scenario immediately. G3B never relaxes provider quality thresholds to force a pass; a recorded rejection is carried forward to G3C as qualification evidence.
