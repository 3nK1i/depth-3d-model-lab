# I6.6B — Worker Progressive Snapshot Emission & Throttling

## Purpose

I6.6B instruments the existing isolated PyCOLMAP sparse-reconstruction worker to emit bounded, provisional sparse snapshots conforming to the frozen I6.6A authority contract while mapping is still running.

This step is deliberately worker-side only. The application backend/controller does not consume the new channel yet; that transport/session-isolation work belongs to I6.6C.

## Observation boundary

The existing `pycolmap.incremental_mapping()` callbacks report registration progress but do not pass the current `Reconstruction` object to the callback. I6.6B therefore reuses PyCOLMAP's native incremental snapshot facility:

```text
PyCOLMAP incremental mapper
    ↓ every bounded registered-view interval
native temporary sparse snapshot
    ↓ next-image callback
bounded dependency-neutral preview extraction
    ↓
EIA_SFM_PROGRESSIVE <JSON>
    ↓
temporary native snapshot deleted
```

The native snapshot files are observation artifacts only. They do not become application persistence or canonical scene state.

## Bounded cadence

The native snapshot interval is deterministic and source-size-aware:

- target approximately 12 mapping snapshots;
- minimum native interval: 2 newly registered views;
- no change to source frame/view planning;
- no retry-until-pass behavior.

A second wall-time throttle requires at least 0.75 seconds between accepted mapping emissions. Native snapshot tokens are deduplicated. A skipped native snapshot is deleted rather than queued indefinitely.

The final selected sparse model is emitted once more as a `finalizing` **provisional** snapshot immediately before the existing conversion/final-result admission path. This guarantees a last preview without promoting it to final authority.

## Bounded geometry

Each progressive payload is bounded to at most:

- 64 sampled registered cameras;
- 4096 sampled sparse landmarks.

Sampling is deterministic and preserves the ordered ID range endpoints. Statistics still report the current full candidate counts; the geometry arrays are previews only.

Camera poses are converted from PyCOLMAP `camera_from_reconstruction` to the existing I6.6A `T_reconstruction_from_camera` convention before emission.

## Transport boundary

I6.6B uses a distinct stdout prefix:

```text
EIA_SFM_PROGRESSIVE
```

The pre-existing worker protocol remains:

```text
EIA_SFM_EVENT
```

No value is added to `SfmWorkerEventType`, and `core/reconstruction/sfm_worker_protocol.py` is unchanged. Consequently, the existing host backend ignores I6.6B progressive lines. I6.6C will add validated parsing, session isolation, stale-revision rejection, and coalescing/backpressure.

The existing worker request schema and final `result.json` schema are unchanged.

## Fail-soft rule

Progressive visualization must never make a scientifically valid final sparse reconstruction fail. Snapshot loading/conversion/emission failures are caught inside the worker, surfaced as ordinary worker warnings, and mapping/final conversion continues.

This does not hide final SfM failures: feature extraction, matching, mapping, model selection, and final result conversion retain their existing failure semantics.

## Authority

Every emitted payload mirrors I6.6A:

- backend authority: PyCOLMAP;
- arbitrary reconstruction gauge;
- provisional visualization only;
- not an admitted final sparse result;
- not persistent canonical scene state;
- not canonical geometry authority;
- not metric scale;
- not measurement authority;
- not certified metrology.

## Scientific non-effects

I6.6B does not change:

- SIFT extraction configuration;
- matching strategy/thresholds;
- incremental mapper thresholds;
- multiple-model policy;
- bundle adjustment configuration;
- final candidate selection;
- I6.5 frame/view request or effective plan;
- provider selection or production Auto;
- dense reconstruction;
- controller/UI/viewer behavior.

## Qualification requirement

Before I6.6B is frozen, closure requires both automated regression and a real PyCOLMAP 4.1.1 worker replay demonstrating:

1. at least one valid `mapping` progressive snapshot on a sufficiently registerable sequence;
2. a valid `finalizing` provisional snapshot;
3. strict increasing revisions accepted by the I6.6A transition validator;
4. bounded geometry samples;
5. unchanged successful final worker result;
6. no leftover `pycolmap_progressive_native` directory after worker exit.

I6.6C is the next step and will add application-side transport, session isolation, stale/out-of-order rejection, and coalescing/backpressure without yet changing the viewport.
