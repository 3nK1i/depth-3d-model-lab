# I6.7C — Dense Transport, Backpressure & Memory Bounds

## Purpose

I6.7C transports the frozen I6.7A provisional dense snapshots emitted by the frozen I6.7B provider wrapper without turning them into admitted or persistent geometry.  The step is deliberately transport-only: no viewport, Reconstruction-page UI, provider selection, production Auto, final dense admission, mesh generation, or calibration behavior is added.

## Dedicated transport channel

I6.7C defines the dedicated line prefix:

```text
EIA_DENSE_PROGRESSIVE <validated I6.7A payload>
```

Unrelated worker/process output is ignored.  Matching lines are parsed through `ProgressiveDenseSnapshot.from_payload`, so the frozen I6.7A authority labels, snapshot digest, concrete provider identity, source-SfM binding, request fingerprint, reconstruction frame, revision ordering, and stage ordering remain authoritative.

`ProgressiveDenseTransportLineSink` implements the I6.7B snapshot-sink interface directly.  It serializes already-validated provisional snapshots without changing provider execution or scientific output.

## Backpressure policy

`ProgressiveDenseSnapshotBuffer` is latest-only:

- at most one pending snapshot exists;
- a newer valid pending snapshot replaces the older pending snapshot;
- revision gaps remain valid by the frozen I6.7A contract;
- stale/duplicate, stage-regressing, cross-run, cross-request, cross-source, cross-provider, and cross-frame updates are rejected;
- malformed/oversized transport values are rejected at the transport boundary;
- delivery state retains only the last delivered snapshot plus the latest pending snapshot.

This is visualization/process protection only.  Coalescing never changes the dense provider's scientific execution, its full geometry artifact, or any final admission decision.

## Memory bounds

I6.7C adds explicit operational transport limits:

- maximum serialized transport snapshot: **4 MiB**;
- maximum I6.7B-compatible preview population: **4096 points**;
- maximum pending snapshot count: **1**;
- maximum retained snapshot count: **2** (`last delivered + latest pending`);
- maximum retained serialized-payload budget: **8 MiB**.

These are transport/viewer-safety limits, not reconstruction quality thresholds.  The provider's full dense artifact remains unchanged and outside the progressive transport payload.

The buffer records accepted/coalesced/rejected/delivered counts plus current/peak pending and retained serialized-payload bytes.

## Frozen authority boundary

I6.7C cannot:

- admit the progressive snapshot as a final dense result;
- create a persistent canonical dense scene;
- establish canonical camera/global-frame authority;
- claim physical metric scale;
- make measurements authoritative;
- claim certified metrology;
- generate a surface mesh;
- promote a dense provider or activate production Auto.

PyCOLMAP SfM remains the sole camera/global reconstruction-frame authority.

## Real qualification

I6.7C does **not** rerun PatchMatch or Stereo Fusion.  Its real qualification replays the immutable successful I6.7B `progressive_dense_snapshots.jsonl` through the new I6.7C line codec and latest-only buffer.

The replay must prove:

1. the JSONL SHA256 matches the frozen I6.7B report;
2. all frozen snapshots parse through I6.7A;
3. the first snapshot can be delivered and all later intermediate snapshots can be coalesced under maximal backpressure;
4. the final I6.7B geometry snapshot survives coalescing exactly by `snapshot_sha256`;
5. no valid frozen snapshot is rejected;
6. pending/retained serialized payload never exceeds 4/8 MiB;
7. final/provisional authority boundaries remain unchanged.

## Deliberately excluded

- no application-controller dense state or QTimer coalescing;
- no incremental dense viewer (I6.7D);
- no sparse-to-dense final handoff/admission (I6.7E);
- no professional dense UX/performance closure (I6.7F);
- no I7 mesh work;
- no I8 Gaussian Splatting;
- no I9 calibration/metric-scale authority.

The application/controller binding is intentionally deferred until the incremental dense viewer step consumes this frozen bounded transport surface.

## I6.7C qualification and formal freeze gate (2026-09-19)

The authoritative `depth` environment reported 16/16 focused tests, 32/32 frozen I6.7B focused tests, 27/27 I6.7A, 179/179 concrete-provider predecessors, 21/21 I6.4 boundaries, 35/35 I6.6 authority boundaries, 10/10 I6.6C transport regressions, and 4596/4596 full-unit tests passing with `I6_7C_CLOSURE_RC=0`. Compilation, shell syntax, import, and scope guards passed. Earlier negative full-unit evidence caused by the host falling below the unchanged 8 GiB reconstruction storage safety floor is retained as negative environmental evidence; the corrected full suite passed after restoring storage headroom. This paragraph records the user's reported terminal results; an immutable closure receipt must additionally bind an archived, successfully executed closure log.

The frozen successful I6.7B production snapshot stream (`progressive_dense_snapshots.jsonl`, SHA256 `d8edbc22927c3682179e19e783007c9a434af743fc9bb3569193fc1e6266d0ba`) was replayed through the I6.7C codec/buffer without rerunning PatchMatch or Stereo Fusion: 23 input snapshots; first revision 1; final revision 23 at `finalization` with geometry; 4096 preview points; 1,686,160 current dense points; 2 accepted, 21 coalesced, 0 rejected, and 2 delivered. Peak pending payload: 704,169 bytes against 4 MiB; peak retained payload: 705,781 bytes against 8 MiB. No final admission, persistence, production Auto, metric-scale claim, or mesh generation occurred; PyCOLMAP camera authority remained preserved.

Closure is **not declared by this document alone**. I6.7C is `CLOSED / FROZEN` and I6.7D may start only when the standalone closure tool verifies the frozen I6.7B source checkpoint and SHA256s, exact I6.7C source hashes, archived automated closure PASS log, and independent evidence replay; creates a unique immutable I6.7C closure/source checkpoint with SHA256 sidecars and a closure receipt; and independently verifies every artifact. Do not use Git/Gitea as a development gate or alter any frozen I6.7A/I6.7B, controller, UI, Auto, final-admission, or scientific parameter boundary.
