# I4.3G4B — Single-Image Production-Native Qualification

I4.3G4B consumes the passing I4.3G4A source/runtime/evidence preflight and produces new real production-native qualification evidence for all six authorized single-image depth providers across the three frozen G4 scenarios.

## Boundary

- execute one frozen scenario at a time;
- reuse the exact G4A-bound G2 authoritative PyCOLMAP SfM JSON and source bytes without rerunning SfM;
- share that same authoritative camera/scene frame across all six providers within the scenario;
- build an independent existing I4.3F adaptive production plan for each provider so execution remains hardware-adaptive rather than imposing a permanent global view cap;
- execute Metric3D v2, UniDepthV2, MoGe-2, Depth Pro, Depth Anything V2 and Marigold sequentially through their existing I4.3E production adapters;
- persist source, SfM, runtime/checkpoint, adaptive-plan, native-worker, canonical-geometry, timing and measured device-wide peak-VRAM evidence with SHA256 identities;
- preserve existing provider quality thresholds unchanged;
- matched-view diagnostics, blinded/manual review and scenario verdicts remain I4.3G4C work.

## Source and camera authority

PyCOLMAP remains the sole authoritative global camera-pose and scene-frame owner. G4B does not rerun SfM and does not accept provider-native poses as canonical. The frozen workspace may contain materialized input keyframes that PyCOLMAP did not register; this is valid SfM state. G4B verifies every materialized input byte, then creates an immutable registered-view-only provider-input projection using the unchanged authoritative PyCOLMAP camera transforms/landmark coordinates before applying the existing I4.3F adaptive plan. The frozen SfM JSON itself is never rewritten. Before and after provider execution, the source asset-set identity and frozen SfM SHA256 are rechecked. A source, runtime, evidence, storage-safety or authority drift is a hard scenario failure.

## Production outcomes

Each provider must reach one terminal production-native outcome: `pass`, `degraded`, or `rejected`.

`rejected` is valid only for the provider's already-existing I4.3E minimum accepted SfM-aligned-view quality gate. It is not a software failure. G4B preserves the native worker artifacts, records the accepted/required counts, proves no threshold change or retry-to-pass occurred, records no canonical geometry for that rejected run, and continues the remaining provider matrix. Every other exception remains a hard scenario failure.

A scenario production snapshot is complete only after all six providers have terminal evidence. A legitimate provider rejection does not make the orchestration snapshot fail; the rejection is carried unchanged into G4C and cannot be overridden by manual review.

## Resource and evidence policy

The G4A free-disk safety floor is reused before scenario preflight and before each provider. Current runtime identities must still match their frozen G4A identities. Peak GPU memory is measured using the existing device-wide `nvidia-smi` sampling method; unavailable optional G1 metrics are never fabricated and instead carry explicit reasons.

No provider is selected or promoted. G4B does not implement Auto, authorize Hybrid, claim physical metric scale, measurement authority or certified metrology, generate an authoritative surface mesh, or introduce a permanent global view cap.

## Execution workflow

Run `scripts/run_i4_3g4b_single_image_scenario.py` in `preflight` mode first for each frozen scenario. Preflight verifies the G4A snapshot and predecessor chain, source/SfM hashes, current storage floor, hardware profile and all six current runtime identities without starting provider inference.

Then run the same scenario in `production-native` mode into a new empty evidence directory. Preserve every attempt; never overwrite a failed or rejected attempt. Execute the three scenarios separately so a genuine scenario failure can be inspected before continuing.

G4B is not formally complete until all three scenario production snapshots have been inspected and accounted for. I4.3G4C remains mandatory for the frozen matched-view/manual review dimensions and formal G4 closure.


## G4B-discovered Metric3D native-confidence correction

The first real Complete Room attempt exposed a historical I4.3E1 assumption that Metric3D v2's native depth-confidence output was already normalized to `[0,1]`. The official ViT/RAFT decoder returns the depth-confidence channel directly from the iterative flow state without a sigmoid or clamp, so that range assumption is not valid.

G4B therefore preserves the finite native Metric3D confidence map and its observed min/max as provider evidence, but does **not** reinterpret, clip, sigmoid, rank-normalize, or otherwise invent a mapping into canonical `[0,1]` fusion confidence. Canonical fusion uses neutral confidence weighting for Metric3D until a separately justified normalization/calibration policy is qualified. Non-finite native confidence remains a hard runtime failure. This correction does not change Metric3D depth, SfM alignment thresholds, camera authority, scale authority, provider status thresholds, or any other provider's confidence semantics.
