# Phase 8.9B — QC Report Export & Qualification

Phase 8.9B productizes **existing Phase 8.7C inspection/QC evidence** inside the Phase 8.9A reconstruction export bundle. It does not add a second QC engine, report schema, persistence path, export subsystem, or backend lifecycle.

## Production scope

- `app/widgets/reconstruction_page.py`
  - accepts the existing A1 `InspectionReportExportSource` as explicit product-export authority;
  - presents the exact same immutable `InspectionReport` through the existing Phase 8.8E4 inspection/QC bridge;
  - includes qualified QC evidence in the existing `Export Bundle` action;
  - keeps display-only E4 reports non-exportable unless the matching A1 authority source is explicitly bound;
  - clears stale QC export qualification when E4 evidence is replaced/cleared;
  - preserves session fail-closed behavior and existing geometry authority boundaries.

No core QC writer changes are required: A1 already writes `qc/inspection_report.json`, records SHA-256/provenance in the reconstruction export manifest, verifies bundle integrity, binds the report to `INSPECTION_REPORT` / `inspection_decision` authority, and rejects certified-metrology claims.

## Safety invariants

- QC remains inspection-decision authority only.
- QC never becomes canonical geometry, measurement, or metrology authority.
- Gaussian/NeRF remain appearance-only.
- `certified_metrology=false` remains mandatory.
- QC is never recomputed by the export UI.
- A serialized/display-only report cannot manufacture export authority.
- Session mismatch blocks export before source binding.
- Existing Save/Reload/checkpoint/recovery behavior is unchanged.
- No heavy reconstruction dependency is imported into the production UI.

## Validation

Run:

```bash
./scripts/run_phase8_9b_tests.sh quick
./scripts/run_phase8_9b_tests.sh affected
```

Manual validation is limited to the affected Reconstruction UI: attach qualified QC evidence in the test path, verify `Product export` reports QC availability, export a bundle, and inspect the generated `qc/inspection_report.json` plus manifest/checksum. Full regression remains a later Phase 8.9 release gate.
