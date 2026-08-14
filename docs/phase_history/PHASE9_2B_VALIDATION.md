# Phase 9.2B — Custom Class / Segmentation Project Foundation

Phase 9.2B adds a reusable, dependency-light segmentation-project contract above the
Phase 9.2A per-frame semantic workspace.

## Scope

- Stable project IDs/names and deterministic JSON persistence.
- Stable YOLO-compatible class IDs/names across frames and sessions.
- Existing model classes can seed a project catalog.
- Custom classes allocate after the highest existing class ID.
- Custom/imported classes may be renamed while preserving their stable class ID.
- Model-defined classes remain immutable; reclassification uses a custom project class.
- Deterministic human-friendly instance names such as `robot_arm_001`.
- Filesystem-backed project store uses SHA-256-derived filenames so project IDs cannot
  escape the store directory.
- Project catalogs can be merged into unlabeled Auto-Segment workspaces without
  touching masks or authority.

## Authority

Segmentation projects and semantic labels are context only. They never become
canonical geometry, metric-scale, measurement, QC, or certified-metrology authority.

## Deferred to later Phase 9.2 steps

- Auto-Segment inference orchestration.
- Interactive class/object editor UI.
- Mask refinement / undo-redo.
- Workspace/project persistence beyond the reusable class-project record.
- Semantic 3D viewer interaction.
