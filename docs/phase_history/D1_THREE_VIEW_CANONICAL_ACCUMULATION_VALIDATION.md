# D1 — Three-View Canonical Accumulation Validation

D1 validates raw multi-view registration before any spatial fusion is allowed to hide disagreement.

The runner reuses the already-validated B2 PyCOLMAP result, selects exactly three registered keyframes from the same globally optimized SfM world, runs isolated DA3 depth inference independently for each view, aligns each depth map through the B1 scale-only bridge, and projects each RGB-D frame through the Phase C canonical projection path.

The three XYZRGB clouds are concatenated only for inspection. D1 does **not** perform voxel merging, multi-view filtering, persistence, metric-scale qualification, measurement authority, or canonical-scene authority promotion.

Default selection divides the registered camera track into temporal thirds while preserving the exact B2/C anchor keyframe in its third. Explicit three-keyframe selection is also supported.

Acceptance requires all three real depth alignments and projections to succeed, followed by visual inspection in both RGB and source-color modes. The source-color preview is specifically intended to expose duplicated or displaced surfaces before D2 numerical overlap diagnostics.
