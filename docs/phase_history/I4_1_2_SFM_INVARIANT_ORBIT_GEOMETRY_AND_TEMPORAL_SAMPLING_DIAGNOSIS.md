# I4.1.2 — SfM-Invariant Orbit Geometry & Temporal Sampling Diagnosis

## Purpose

I4.1.2 is diagnosis-only. It was added after the Horse contact sheet showed a visibly broad outside-looking-in trajectory while I4.1/I4.1.1 reported very small orbit coverage, and a longer clip still failed dense consistency.

The milestone separates two possible causes without changing any acceptance policy:

1. **temporal representation** — whether the 40-keyframe cap represents the complete source clip or only its prefix;
2. **orbit geometry** — whether coverage remains stable under coordinate-invariant plane and target estimators.

## Evidence

The runner reuses the existing I4.1 directory and reads only saved artifacts. It accepts either the `object_orbit_i1_baseline` stop or the later `object_orbit_baseline_quality` stop.

It reports:

- source duration estimated from reported frame count / source FPS;
- sampled timestamp span and last sampled source-frame fraction;
- whether the 40-frame cap likely truncates the source to a prefix;
- registered / aligned / selected camera sets when available;
- sparse-landmark median, robust optical-axis consensus, trajectory-circle center, and current Core-median target when available;
- target-through PCA, camera-centered PCA, and direction-PCA orbit coverage;
- target-independent camera-trajectory spread;
- exact reproduction of the current I4 metric when Core evidence exists.

## Authority

I4.1.2 does **not** rerun or modify SfM, DA3, alignment, dense consistency, fusion, F, G, metric scale, measurements, mesh, thresholds, sampling policy, or orbit policy. Any policy-correction candidate remains explicitly unauthorized until a later reviewed milestone.

## Expected next decision

If the saved workspace reaches the 40-frame cap while representing only a prefix of a longer video, the next step should be a sampling-policy correction that distributes keyframes over the full source duration before rerunning expensive I4 qualification. Thresholds must not be relaxed to compensate for incomplete temporal coverage.
