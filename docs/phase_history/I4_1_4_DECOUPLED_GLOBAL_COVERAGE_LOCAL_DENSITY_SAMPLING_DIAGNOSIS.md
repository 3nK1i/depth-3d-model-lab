# I4.1.4 — Decoupled Global-Coverage / Local-Density Sampling Diagnosis

## Purpose

I4.1.3 corrected long-video keyframe sampling so a capped 40-frame SfM set spans the complete source. On the Horse orbit this improved DA3/SfM alignment to 36/40, but the dense-consistency path still collapsed to two views. I4.1.4 determines whether that collapse is primarily caused by reduced local temporal density, by the capture/content regime itself, or by both.

## Evidence compared

The diagnosis can compare three existing runs without rerunning reconstruction:

- current full-span Horse run after I4.1.3;
- same-source long Horse run before I4.1.3 (2 FPS prefix sampling);
- earlier 30-second Horse control segment.

For every logged dense-consistency edge it combines the exact pass/degraded status with SfM workspace timestamps and camera poses, then reports pass fraction against temporal spacing, order gap, camera baseline/standoff, target-centered viewpoint angle, and source segment.

## Authority

Diagnosis only. It does not rerun or alter SfM, DA3, pair qualification, fusion, F/G, sampling policy, dense-consistency thresholds, metric scale, measurement authority, or mesh state. I4.2 remains unauthorized.

## Decision rule

A decoupled global/local sampling architecture is only a candidate when evidence shows materially better dense-pair qualification at higher local density. A cleaner static capture is recommended when the same-source dense prefix remains poor and a different short control segment is materially better. Threshold relaxation is never authorized by this milestone.
