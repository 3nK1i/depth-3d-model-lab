# D4.3 — Repaired Ten-View Chain Qualification

D4.3 is the minimal repair gate after D4.2 classified the single weak D4.1 transition as pair-specific incompatibility.

It preserves the existing D4 views, runs DA3 only for the immediately preceding registered keyframe, constructs two ten-view candidates by removing either weak-pair member, and recomputes all nine adjacent bidirectional reprojection/depth-consistency checks under the unchanged D4.1/D2 thresholds.

No thresholds are relaxed. No voxel fusion, multi-view filtering, persistence, geometry authority, metric scale, measurement authority, or certified metrology is introduced.

A passing D4.3 result requires at least one complete repaired ten-view candidate to pass all nine adjacent links. If both pass, the candidate with the stronger worst-case dense consistency is selected. The selected cloud remains raw/unfused and exists only to authorize the next fusion milestone.
