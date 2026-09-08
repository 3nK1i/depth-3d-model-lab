# I5.7C - RollingDepth Robustness Qualification + Closure

## Scope

I5.7C is evidence-only. It consumes the frozen I5.7A admission and the exact successful I5.7B real replay. It does not run RollingDepth, PyCOLMAP SfM, MVS, DA3, Video Depth Anything, MP-SfM, Hybrid, or Auto.

## Frozen I5.7B conclusion

The current frozen-foundation RollingDepth replay passed all six admitted Castle scenarios:

- castle_control (recovered from the preserved first real attempt without provider rerun)
- castle_low_overlap
- castle_weak_parallax
- castle_overexposure
- castle_symmetry
- castle_dynamic_proxy

The recovered castle_control wrapper-level wall-time and resource observations remain explicitly missing because the original wrapper failed after provider execution but before scenario-record persistence. No performance metric is fabricated and castle_control is not rerun.

## Historical temporal evidence remains active

I5.7C does not overwrite earlier production-native temporal evidence:

- g3_temporal_static_sweep: DEGRADED
- g3_temporal_motion: REJECTED
- g3_temporal_low_texture: PASS

Therefore the six-scenario I5.7B PASS qualifies the current production RollingDepth path only within the admitted current Castle temporal envelope. It does not establish universal RollingDepth robustness.

## Preserved authority and non-claims

PyCOLMAP remains the sole camera/scene-frame authority. RollingDepth is not promoted to pose, metric-scale, measurement, metrology, or mesh authority. No provider ranking, universal provider winner, global scalar score, threshold relaxation, production-default mutation, MP-SfM, Hybrid, or Auto policy is introduced.

## Remaining limitations

I5.7C does not claim a complete runtime-performance or resource envelope. The recovered control lacks wrapper-level wall-time/VRAM/RAM observations. RTX 4090, RTX 5090, cross-GPU equivalence, long-sequence resource robustness, provider-specific cancellation, ground-truth dense depth, ground-truth surface accuracy, metric scale, and certified metrology remain unqualified.

## Closure decision

I5.7 closes with a bounded PASS envelope and authorizes I5.8 Video Depth Anything robustness. I5.8 is not started by I5.7C.
