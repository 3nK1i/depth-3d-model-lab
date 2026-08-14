# Phase 9.2F — Deterministic Multi-Mask Editing / Merge

## Scope

Phase 9.2F adds deterministic multi-mask union editing to the existing Auto Segment workspace.
It does not run another model, does not change measurement/reconstruction authority, and does not yet connect reviewed masks to Analyze Frame.

## Behavior

- Check two or more proposal masks in the Proposals list.
- The canvas highlights masks selected for merge.
- `Merge Selected` unions the dense boolean masks exactly.
- The current selected object is the primary object when it is part of the merge.
- A primary assigned class/instance is preserved.
- Source proposals are replaced by one merged proposal.
- The merged result always returns to `Pending` review.
- Structured source object/mask IDs are retained as merge provenance.
- Nested merges flatten provenance back to the original source objects/masks.
- The merge mask source is `semantic_merge` and remains semantic-only.

## Authority

A merged semantic mask is object-membership context only. It never establishes canonical geometry, metric scale, measurement validity, QC authority, or certified metrology.

## Deferred

- Reviewed masks as direct Measure & Inspect / Analyze Frame perception input: Phase 9.2G.
- Workspace/mask persistence and reload: later Phase 9.2 persistence step.
