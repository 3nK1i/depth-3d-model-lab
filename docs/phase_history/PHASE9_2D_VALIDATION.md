# Phase 9.2D — Professional Interactive Auto Segment Workspace UI

## Scope

Phase 9.2D adds presentation/process orchestration only. The 9.2A semantic workspace, 9.2B segmentation project, and 9.2C Auto Segment engine remain authoritative core contracts.

The Measure & Inspect sidebar now exposes **Open Auto Segment**. The workspace supports an existing captured frame or an imported image, isolated SAM2 proposal generation, mask-based selection, stable project classes, custom class creation, instance naming, and pending/accepted/rejected review state.

## Authority

Auto Segment masks and semantic labels remain context only. They do not establish canonical geometry authority, metric scale, measurement authority, inspection/QC authority, or certified metrology.

## Runtime safety

SAM2 generation runs through an isolated `QProcess`; the GUI thread does not own inference. Existing long-running-job/reconstruction guards remain authoritative, and Live 3D model resources are released before the external Auto Segment job starts.

## Deferred to later 9.2 steps

Prompt/refinement and brush-based mask refinement, semantic workspace persistence, downstream Measure & Inspect consumption, and final Phase 9.2 closure remain out of 9.2D scope.

## Candidate correction

The UI loader preserves the worker manifest's stable semantic `object_id` independently from the mask's `mask_id`. This keeps proposal selection/review IDs stable across the isolated job boundary while retaining the original mask identifier for mask provenance.
