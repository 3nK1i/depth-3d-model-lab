# Phase 9.2A — Semantic Workspace Contracts

Phase 9.2A establishes the dependency-light semantic review contracts for the later
Custom Segmentation / Auto Segment workspace. It does not add the interactive UI,
new inference backends, persistence, training, or 3D viewer features.

## Architecture

- Existing YOLO/SAM perception contracts remain authoritative for detections/masks.
- `ObjectObservation` remains the downstream compatibility contract.
- A `SemanticWorkspace` adds review state above existing observations without
  duplicating or mutating mask payloads.
- Model classes and custom project classes share a YOLO-compatible `class_id` /
  `class_name` catalog.
- Class identity and instance identity are separate (`pallet` vs `pallet_001`).
- Auto-Segment mask proposals may remain genuinely unlabeled until the user assigns a class; no placeholder/fake semantic label is invented.
- Accepted reviewed objects normalize back to `ObjectObservation` so Measure &
  Inspect and Phase 8 semantic 3D geometry can reuse the existing pipeline.
- Semantic labels/masks are context only: never canonical geometry, metric-scale,
  measurement, QC, or certified-metrology authority.

## Explicitly deferred

- Auto Segment engine/orchestration
- click-to-select / class naming UI
- mask refinement/editing UX
- semantic project persistence
- application-wide workspace integration
- 3D semantic label/box presentation

These belong to later Phase 9.2 / Phase 9.3 substeps.
