# I6.6A — Progressive SfM Snapshot & Authority Contract

## Purpose

I6.6A freezes the dependency-neutral contract for presenting **provisional sparse PyCOLMAP state while reconstruction is still running**. It does not emit snapshots from the worker yet and does not modify IPC, the controller, the viewport, or the Reconstruction UI.

The purpose is to establish an authority-safe payload before I6.6B instruments the worker and before I6.6C/D transport/render any progressive geometry.

## Authority model

The lifecycle is explicitly:

```text
PyCOLMAP running
    ↓
provisional snapshot revision 1
    ↓
provisional snapshot revision N
    ↓
worker completes
    ↓
existing I6.2 final-result admission boundary
    ↓
admitted sparse result
```

A progressive snapshot is therefore **not**:

- an admitted `OfflineSfmResult`;
- a persistent canonical scene;
- canonical geometry authority;
- metric-scale evidence;
- measurement authority;
- certified metrology evidence.

PyCOLMAP remains the sole camera/global-frame authority. Snapshot poses use the existing `T_reconstruction_from_camera` convention in arbitrary monocular reconstruction gauge.

## Snapshot identity

Each `ProgressiveSfmSnapshot` carries:

- exact `session_id`;
- positive monotonic `revision`;
- UTC timestamp and monotonic elapsed time;
- stage (`mapping` or `finalizing`) and stage detail;
- canonical PyCOLMAP backend identity/version;
- exact worker-request SHA-256 fingerprint;
- provisional sparse statistics;
- optional candidate-model identity/count;
- optional bounded camera/landmark preview samples;
- warnings;
- immutable authority flags;
- deterministic `snapshot_sha256`.

The snapshot digest is a provenance/deduplication identity. It is not a quality score and does not rank reconstruction candidates.

## Statistics

The provisional statistics are kept separate from final `SfmStageStatistics` and currently record:

- input views;
- registered cameras in the current extracted PyCOLMAP candidate;
- sparse landmarks;
- sparse observations;
- optional mean reprojection error.

These numbers may change between revisions. I6.6A intentionally does **not** require registered-camera or landmark counts to increase monotonically because multi-model mapping and bundle-adjustment cleanup can change the current provisional visualization candidate.

## Optional geometry preview

`ProgressiveSfmGeometryPreview` may carry dependency-neutral bounded samples:

- camera samples: backend image/camera IDs, source image name, `T_reconstruction_from_camera`;
- landmark samples: backend point ID, arbitrary-gauge reconstruction position, optional RGB and track length.

I6.6A deliberately does not choose operational sampling limits or throttling cadence. Those are I6.6B responsibilities. The contract only guarantees unique sample identities and requires sample counts to stay within the reported provisional totals.

## Revision / stale-update semantics

`validate_progressive_sfm_snapshot_transition()` requires:

- same session;
- same exact worker request fingerprint;
- same backend identity/version;
- same input-view count;
- strictly increasing revision;
- non-decreasing elapsed time / UTC timestamp;
- no stage regression from `finalizing` back to `mapping`.

Revision numbers **do not have to be contiguous**. This is intentional so I6.6C may later coalesce/backpressure updates and drop superseded intermediate snapshots without treating the gap as corruption.

## Fail-closed authority parsing

Serialized payloads are parsed fail-closed. Contradictory authority labels, non-boolean authority flags, altered snapshot digests, non-PyCOLMAP backend identity, completed (`1.0`) progress, or attempts to claim final/persistent/metric/metrology authority are rejected.

## Non-effects

I6.6A does not:

- modify `scripts/reconstruction_pycolmap_worker.py`;
- emit progressive snapshots;
- modify the existing SfM worker protocol;
- change feature extraction, matching, mapping, BA, retry, or scientific thresholds;
- modify `ReconstructionController`;
- modify the Reconstruction page or VisPy viewport;
- create a persistent canonical scene;
- run dense reconstruction;
- alter provider selection or production Auto;
- replan I6.5 frame/view intent while SfM is running;
- add progressive dense visualization (I6.7 scope);
- claim metric scale or certified metrology.

## Next

I6.6B may instrument the isolated PyCOLMAP worker to extract bounded snapshots under an explicit throttling/coalescing policy while preserving the exact I6.6A provisional-authority contract and the existing final-result admission path.
