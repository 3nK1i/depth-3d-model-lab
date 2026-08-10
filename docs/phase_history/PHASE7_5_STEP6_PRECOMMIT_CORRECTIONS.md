# Phase 7.5 Step 6 — Pre-commit corrections

This correction is intended for the uncommitted Step 6 working tree.

It fixes three findings from the real end-to-end validation:

1. Volume is now rejected when width, height, or depth is unavailable or rejected.
2. Per-object X/Z diagnostic plots draw each 3D box as a convex projected silhouette, avoiding misleading self-crossing wireframes. The aligned RGB report continues to render the complete 12-edge projected 3D boxes.
3. `measurements.json` now includes a stable `object_index` for every serialized object.

No Qt UI, depth worker, perception worker, reconstruction feature, model adapter, or application version is changed.
