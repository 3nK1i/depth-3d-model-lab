# Phase 9.6E2 — Full-Scene DA3 Recorded-Video Workflow

## Scope

E2 adds a recorded-video-only **Build Full Scene** action inside the existing collapsed **Build & Processing** group. The previous **Integrate DA3** action remains unchanged as the bounded Quick path.

The Full Scene workflow:

1. scans the complete recorded video;
2. applies conservative frame-quality filtering and adaptive temporal/view-change selection;
3. bounds the selected set to at most 96 keyframes by default;
4. runs DA3-SMALL in the isolated `eia-da3` runtime using resident bounded 3-view windows with 2-view overlap;
5. aligns successive windows into one relative/arbitrary reconstruction gauge using the shared camera poses plus robust overlapping depth ratios;
6. consolidates the supporting DA3 evidence and reuses the existing canonicalization + persistent-scene path;
7. admits the resulting persistent scene only after the generic E1 job/output identity checks and persistent-file integrity verification.

## Authority / truthfulness

- canonical geometry remains the only geometry/picking/measurement authority after canonicalization;
- DA3 window output is supporting relative geometry only;
- `unit_to_metre` remains unavailable;
- no metric-scale, measurement, QC, or certified-metrology authority is claimed;
- E2 sequential overlap does **not** claim loop closure;
- E2 full-scene quality is not certified until the later real-video qualification campaign.

## Resource policy

The production `depth` process does not import DA3. One external DA3 process loads the model once and sequentially processes bounded three-frame windows. The whole child tree remains inside the E1 isolated process group so Cancel/shutdown uses the already-qualified bounded cleanup path.

## Validation

Run:

```bash
./scripts/run_phase9_6e2_tests.sh quick
./scripts/run_phase9_6e2_tests.sh integration
./scripts/run_phase9_6e2_tests.sh backward
```

Real manual qualification is intentionally limited in E2 to one representative recorded-video build on the available RTX 2060. Do not stage generated video/keyframe/output artifacts.


## Manual-runtime UI correction

A real 1366×768 manual attempt exposed a presentation-only issue before the E2 job started.
The BUILD & PROCESSING action row used horizontal `Ignored` size policies separated by a stretch,
which allowed both `Build Full Scene` and `Cancel build` to collapse to zero width in the narrow
settings sidebar. The user therefore invoked the unrelated top-level Offline Reconstruction `Start`
action; its `Progress must be monotonic...` failure belonged to that legacy offline workflow, while
the E1 processing supervisor remained Idle and no E2 job had started.

The correction keeps both E2 action buttons visible with equal expanding shares at the qualified
1366×768 workspace width and makes the ready-state text explicitly direct the user to `Build Full Scene`.
No offline reconstruction, DA3 runtime, canonicalization, authority, or processing protocol behavior is changed.
## Real-video consolidation correction

The first full-scene TUM run completed all external DA3 windows and all overlap
similarity estimates, but final consolidation failed because raw DA3 windows
carry OpenCV W2C extrinsics as 3x4 matrices while transformed later windows are
materialized as homogeneous 4x4 matrices. E2 now normalizes every chunk to a
single homogeneous 4x4 representation before accumulation. This is a
representation-only correction: it does not alter pose values, arbitrary-scale
authority, alignment policy, loop-closure claims, or canonicalization rules.
The multi-window integration fixture now uses real DA3-shaped 3x4 extrinsics so
this regression is covered directly.


## Real-video intrinsics consolidation hardening

The TUM long-office replay reached canonicalization after all 85 DA3 windows aligned, but the unchanged F3 camera-profile guard correctly rejected raw per-window DA3 focal jitter (`maximum_relative_intrinsics_spread=0.222591 > 0.10`). The source is one fixed recorded-video camera; principal point was invariant and fx/fy varied coherently, so E2 now treats the variation as DA3 prediction jitter only after a fail-closed fixed-camera admission check.

E2 preserves the raw first-seen per-frame DA3 intrinsics in `raw_intrinsics_per_frame`, records raw focal/aspect/principal-point spread in result provenance, and emits one robust-median *estimated, not calibrated* camera matrix for every consolidated frame. The qualified downstream F3 `0.10` guard is unchanged. The workflow still claims no calibration, no metric scale, no loop closure, no quality certification, and no metrology certification. Captures with inconsistent focal/aspect/principal-point evidence continue to fail closed; users must record without zoom.
