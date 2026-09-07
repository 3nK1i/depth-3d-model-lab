# I5.6B - Current Frozen-Foundation MVS Robustness Replay

I5.6B executes the existing production-native COLMAP MVS adapter sequentially on the six Castle-P19 authoritative SfM cases admitted and hash-bound by I5.6A. It rehydrates frozen PyCOLMAP SfM results and does not rerun SfM or bundle adjustment.

The replay preserves production adaptive view selection, geometric consistency, current cache/image defaults, GPU 0, and `permanent_global_view_cap=None`. Each scenario is attempted once. A structured provider failure is retained as `rejected` evidence and does not authorize easier parameters or retry-until-pass behavior.

Native evidence includes selected views, geometric depth/normal map coverage, fused PLY/native visibility correspondence, visibility-support statistics, wall time, process/GPU resource evidence, adaptive-plan provenance, and pre/post authoritative input integrity. Cross-run absolute geometry, metric scale, ground-truth surface accuracy, provider ranking, Auto, Hybrid, MP-SfM, DA3 reruns, and post-hoc scalar quality thresholds remain outside this step.

I5.6C may interpret the completed replay evidence and close the MVS robustness envelope. I5.7 is not authorized until I5.6C closes I5.6.
