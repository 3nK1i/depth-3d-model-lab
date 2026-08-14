# Phase 9.2H — Workspace Persistence & Safe Class Management

## Scope

Phase 9.2H makes Auto Segment review state durable per segmentation project and adds professional class management without changing geometry/metric/measurement authority.

- Project catalogs remain filesystem-backed and stable across sessions.
- The latest reviewed workspace persists exact source BGR, dense masks, class/instance identity, review state, merge provenance, and mask provenance.
- An exact source-image SHA-256 guards automatic resume; users can also explicitly **Restore Saved**.
- **Manage Classes** shows class source, current-workspace usage, and stable class ID.
- Model-origin classes are protected from rename/delete.
- Custom/imported classes can be renamed with stable IDs.
- Deleting a used custom class explicitly returns its objects to **Unlabeled / Pending** before retirement.
- Retired stable class IDs are persisted and never silently reused.
- A resumed accepted workspace remains compatible with the Phase 9.2G reviewed-mask Measure & Inspect handoff.

## Authority

Semantic persistence remains context only. It never creates canonical geometry authority, metric scale, measurement authority, QC authority, or certified metrology.

## Validation

Run:

```bash
./scripts/run_phase9_2h_tests.sh quick
./scripts/run_phase9_2h_tests.sh integration
./scripts/run_phase9_2h_tests.sh backward
python scripts/run_phase9_2h_workspace_persistence_smoke.py
```

Manual validation is limited to changed behavior: create/accept reviewed masks, close/reopen the Auto Segment workspace and confirm exact-source restoration; open **Manage Classes**, verify model classes are protected, rename a custom class, then delete an unused or used custom class and verify used objects return to Unlabeled/Pending. Finally confirm a restored accepted mask still reaches Measure & Inspect through the 9.2G reviewed-mask path.
