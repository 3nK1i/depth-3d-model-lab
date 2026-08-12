# Phase 8.9A2 — Reconstruction Export UI

## Scope

Phase 8.9A2 wires the already-qualified Phase 8.9A1 authority-aware export contract into the existing Reconstruction workspace. It does not add a backend, persistence format, checkpoint path, QC engine, or new top-level UI mode.

The UI remains inside the existing **SESSION & PERSISTENCE** disclosure group so the Phase 8.8 information architecture and 1366×768 vertical-only sidebar contract remain unchanged. The row is explicitly labelled **Product export** to keep it distinct from **Save Scene**.

## Behavior

- A completed, qualified offline sparse-SfM result can be exported as supporting geometry with arbitrary scale.
- A canonical mesh is exportable only when an already-materialized `CanonicalSurfaceMeshExportSource` is explicitly bound to the page.
- Loading a persistent scene does not manufacture a surface mesh from metadata; persistent manifests do not contain the full mesh arrays.
- Export is blocked while reconstruction is active or when materialized evidence does not resolve to one non-conflicting session.
- Existing export directories are never deleted or replaced. The UI allocates a fresh sibling directory and A1 performs the atomic bundle write.
- Every successful UI export is immediately verified through `verify_reconstruction_export_bundle`.
- Gaussian/NeRF remain appearance-only and are not exported by A2.
- QC report productization remains Phase 8.9B.
- Certified metrology remains false.

## Validation

Run:

```bash
./scripts/run_phase8_9a2_tests.sh quick
./scripts/run_phase8_9a2_tests.sh affected
```

`quick` covers the A2 static contract, A1 export contract, and A2 Qt UI behavior. `affected` additionally re-runs the Phase 8.8 progressive-disclosure/full-UX contracts and whole-workflow UI sanity that A2 must preserve.

Manual validation is intentionally limited to the affected Reconstruction export interaction at 1366×768 plus one tiny backward sanity check. Full regression remains a later Phase 8.9 release gate.
