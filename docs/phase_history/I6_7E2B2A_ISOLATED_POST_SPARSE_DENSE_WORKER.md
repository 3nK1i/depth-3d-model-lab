# I6.7E2B2A — Isolated post-sparse dense execution, first production handoff increment

## Authoritative predecessor

I6.7D is frozen; I6.7E1 verified the complete 1,686,160-point real COLMAP-MVS artifact; I6.7E2A independently qualifies a completed provider result against a **same-session admitted One Start sparse result**; I6.7E2B1 atomically admits the resulting full artifact and requires revalidation for consumption. The frozen I6.7B/C/D provisional preview stays capped at 4,096 points and has no final-admission authority.

## Scope, exact current deliverable

This increment adds the **worker-side** path for *one explicitly selected COLMAP-MVS provider* **after** a successful admitted One Start sparse result. The application-layer worker callable rereads the sparse result and persisted SfM, checks session, PyCOLMAP camera/gauge authority, source evidence and provider identity, requires a previously unused per-run UUID, and executes the unchanged existing external COLMAP-MVS provider through the frozen I6.7B progressive wrapper. Any provisional geometry is emitted by the frozen bounded I6.7C line protocol and is **not** admission evidence. After the provider returns, the worker hashes the independently completed full `fused.ply`, then invokes E2A candidate qualification and E2B1 single-commit final admission/read-back. All full-artifact I/O happens in the isolated worker, not the Qt render thread.

The standalone script accepts only an explicit COLMAP-MVS request with an affirmative flag. It emits structured start/failure/completed records, together with the existing bounded provisional dense lines. It must eventually be launched and supervised by the application controller; it is **not** a new required user-facing CLI workflow.

## Safety boundaries

- No production Auto/fallback/provider ranking changes; other nine providers are not silently admitted by the first COLMAP-MVS path.
- PyCOLMAP sparse SfM owns camera poses and its arbitrary gauge remains nonmetric. No metrology/mesh/persistent-scene claim.
- Sparse source output and the dense run UUID are bound before provider execution. An existing or failed dense-run directory is never overwritten or retried until success.
- The final admission is based only on the complete provider PLY, not progressive snapshots, and is rechecked by the existing E2A/E2B1 gates.
- Real MVS execution was **not** run in this increment: tests use a fake external worker to exercise the same wrapper and final admission, including preview-sink failure, foreign identity and no-overwrite rejection.

## What remains before the normal One Start product is connected

**I6.7E2B2B** must implement qualified, session-bound Qt process ownership, explicit manually selected provider launch after sparse admission, bounded parsing of worker events, cancellation that also terminates the nested MVS subprocess, stale-run rejection, and independent revalidation of completed admission *before* emitting a UI success signal. The current existing `ExternalMvsWorkerRunner` launches a child subprocess without yet proving complete nested-process cancellation for the new product lifecycle; do not enable the real production worker from the UI or run a new heavy provider until that safeguard is demonstrated. No UI button, normal One Start dense execution, or full-resolution viewer is delivered by E2B2A.

**Full-density GPU viewing** then consumes only an independently verified admitted artifact, with measured 1.69M-point rendering, bounded CPU/GPU memory, responsive navigation, and a separate final geometry presentation path. The 4,096-point provisional preview remains untouched.

## Validation

Focused: worker-side fake external provider, explicit opt-in, foreign session/SfM/frame/environment, failed sparse result, preview fail-soft, no-overwrite, plus core/app architecture guard. Predecessor: E2A/E2B1 and frozen dense contract/transport tests. Workstation closure: full unit suite, storage preflight, source-anchor/compile/shell checks. No real SfM, PatchMatch or Stereo Fusion rerun and no Git/Gitea/GitHub operations.
