# I5.5B - Current Frozen-Foundation Dynamic-Scene Replay

I5.5B materializes the exact paired experiment admitted by I5.5A and runs the unchanged production-native PyCOLMAP sparse-SfM path on both cases.

The control is the frozen 19-view Castle-P19 source. The stress case uses the same decoded source pixels plus the frozen source-derived patch translating independently across views. This is a **controlled dynamic-foreground proxy**, not natural dynamic-scene ground truth and not evidence of universal real-world dynamic-scene robustness.

The replay revalidates source identity, lossless control/stress materialization, donor-patch identity, target trajectory, confinement of changed pixels, and whether the moving region is feature-bearing. Control and stress use the same `efficient` profile, exhaustive matching, 19-view scope, and production PyCOLMAP runner. Results report registration, landmarks, observations, reprojection error, mean track length, and sparse-component count. Per-run SfM gauge remains arbitrary, so cross-run absolute geometry/translation comparisons are forbidden.

No dynamic masking, semantic segmentation, motion-outlier rescue, threshold relaxation, production-default modification, dense-provider rerun, MP-SfM, Auto, or Hybrid is introduced. PyCOLMAP remains the sole authoritative camera/scene-frame owner.

A completed replay authorizes only **I5.5C - Dynamic-Scene Handling Robustness Qualification + Closure**. Runtime quality may be PASS, DEGRADED, or REJECTED while the evidence-collection contract itself completes successfully.
