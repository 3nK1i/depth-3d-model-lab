# I4.1.3 — Full-Clip Uniform Keyframe Sampling Correction

## Purpose

I4.1.2 proved that capped `TARGET_FPS` sampling could silently represent only a prefix of a long recorded video. On the Horse orbit case, a ~50.6 s source sampled at 2 FPS with `max_keyframes=40` covered only ~19.5 s (~38.6%) before the cap stopped selecting frames.

I4.1.3 corrects that finite-video behavior without changing SfM, DA3, dense-consistency, filtering, orbit, metric, measurement, or mesh thresholds.

## Policy

- If the requested TARGET_FPS sample count fits within the cap, preserve historical timestamp-driven TARGET_FPS sampling exactly.
- If the finite source would exceed the cap, resolve a deterministic `uniform_full_span` plan.
- Uniform targets include the first and last reported source frame and distribute the capped count across the complete source span.
- Invalid target frames are skipped to the first following valid frame; the sampler never pulls later targets backward.
- EVERY_FRAME and FIXED_INTERVAL behavior is unchanged.

For a ~50.6 s / 29.97 FPS source with requested 2 FPS and cap 40, the effective capped rate is approximately 0.77 FPS and the planned 40 targets cover the full source instead of the first ~19.5 s.

## Provenance

`ReconstructionIngestionResult` now records the resolved sampling plan. The real PyCOLMAP smoke report/summary also records the sampling plan and actual sampled source-span evidence.

A lightweight preflight command can validate a real video before expensive reconstruction:

```bash
python scripts/run_i4_1_3_sampling_preflight.py \
  --video /path/to/video.mp4 \
  --target-fps 2 \
  --max-keyframes 40
```

## Validation policy

Run:

```bash
bash scripts/run_i4_1_3_tests.sh focused
bash scripts/run_i4_1_3_tests.sh quick
bash scripts/run_i4_1_3_tests.sh integration
bash scripts/run_i4_1_3_tests.sh stress
bash scripts/run_i4_1_3_tests.sh affected
```

The new coverage validates long-video full-span behavior, unchanged short-video behavior, deterministic target generation, monotonic unique targets, provenance, real recorded-video preflight, integration with the ingestion pipeline, and stress behavior over randomized/large source sizes.

## Authority

I4.1.3 changes only keyframe sampling policy resolution for capped finite TARGET_FPS sources. It does not authorize I4.2, metric scale, measurements, or surface mesh generation.
