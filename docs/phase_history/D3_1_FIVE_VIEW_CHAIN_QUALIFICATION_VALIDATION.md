# D3.1 — Five-View Chain Qualification

D3.1 qualifies one raw, unfused D3 five-view chain before widening to 10 views.

The gate reuses the validated D2 bidirectional reprojection/depth-consistency diagnostic on all four adjacent pairs. The chain passes only when every adjacent pair passes under the unchanged D2 thresholds.

This step is validation-only. It does not fuse, filter, persist, or promote geometry. SfM scale remains arbitrary and no metric/measurement authority is claimed.

Recommended real candidate after the D3 12→16 result is 11→12→13→14→15, because 13↔14 and 14↔15 are already qualified and this candidate avoids the known borderline 15↔16 pair.
