# I5.4B — Current Frozen-Foundation Forward-Motion / KITTI Replay

I5.4B executes the exact 60-frame contiguous window admitted by I5.4A through the existing production-native PyCOLMAP sparse-SfM path. Matching remains sequential with overlap 10; the source window is not retuned after results are observed.

Forward-motion semantics are promoted from operator-confirmed proxy to authoritative replay evidence only if reconstructed consecutive camera-center motion is forward-axis dominant in the preceding camera coordinate frame. The replay also reports the established I2.4 scale-invariant maximum supporting-pair parallax and baseline-to-range diagnostics. Absolute metric scale is not claimed.

A recognized production ingestion rejection is preserved as structured rejected evidence rather than treated as an infrastructure failure. The known PyCOLMAP outcome `incremental mapping produced no sparse model` is likewise preserved as a distinct incremental-mapping rejection after feature extraction/matching, with reconstruction attempted but no authoritative camera model produced. Any other missing-report failure remains fatal. I5.4B never relaxes thresholds, changes production defaults, runs dense providers, starts MP-SfM, implements Auto, or authorizes Hybrid.

I5.4B may finish PASS, DEGRADED, or REJECTED as a runtime-quality result; valid completed evidence authorizes I5.4C qualification/closure without retry-until-pass behavior.
