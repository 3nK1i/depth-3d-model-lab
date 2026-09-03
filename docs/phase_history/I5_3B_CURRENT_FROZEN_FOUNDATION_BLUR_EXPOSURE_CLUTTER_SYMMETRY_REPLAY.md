# I5.3B — Current Frozen-Foundation Blur / Exposure / Clutter / Symmetry Replay

I5.3B executes the real sparse-SfM replay authorized by the frozen I5.3A admission. The exact Castle-P19 control membership and the five admitted recipes are reused without result-driven tuning or retry-until-pass behavior.


## Pre-SfM ingestion rejection

A stress scenario may legitimately fail the existing production ingestion quality gate before PyCOLMAP mapping starts. The exact production rejection `Fewer than three valid keyframes survived ingestion` is preserved as structured `rejected` evidence, with `sfm_executed=false`; it does not trigger threshold relaxation, recipe tuning, or an infrastructure retry. Unexpected missing-report failures still abort the replay.

## Execution contract

- Materialize control, blur, underexposure, overexposure, clutter, and symmetry from the same frozen ordered source.
- Use lossless PNG staging for every scenario so transformed cases do not receive an extra lossy JPEG recompression confound. The control staging preserves the decoded source pixels exactly.
- Revalidate the intended image-domain stress direction before interpreting robustness:
  - blur: median Laplacian variance decreases;
  - underexposure: median luma decreases and exposure/clipping burden does not improve;
  - overexposure: median luma increases and exposure/clipping burden does not improve;
  - clutter: bottom-band ORB feature density increases;
  - symmetry: mirrored side-band similarity increases.
- Clutter and symmetry remain controlled proxies and are **not natural-scene ground truth**.
- Run the existing production-native `run_reconstruction_pycolmap_smoke.py` path with exhaustive matching for control and all five stress cases.
- Preserve PyCOLMAP as the sole camera/scene-frame authority. Per-scenario monocular gauge remains arbitrary; no cross-scenario absolute metric-scale comparison is permitted.
- Preserve partial registration, degraded results, rejected results, and failed semantic stress revalidation as evidence. Do not tune recipes after observing results.

## Guardrails

There is no threshold relaxation, no production-default modification, no dense-provider replay, no provider ranking/promotion, no metric-scale or metrology claim, no surface-mesh claim, no MP-SfM work, no Hybrid authorization, and no Auto policy work in I5.3B.

A completed I5.3B evidence record authorizes only **I5.3C** qualification and closure. **I5.4** forward-motion/KITTI robustness is not started or authorized by this replay alone.
