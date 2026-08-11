# Phase 8.8E2 — Reconstruction Measurement UI

## Scope

E2 adds a compact reconstruction measurement presentation for the active E1 object target.
It **reuses already-computed Phase 7.5 measurement evidence** through the dependency-light
`InspectionObjectSummary` / inspection-bundle payload shape. E2 never runs perception, depth,
object extraction, fitting, measurement computation, uncertainty evaluation, inspection/QC,
registration, or reference comparison.

The caller must explicitly establish correspondence between the selected persistent semantic
object and the source measurement observation. E2 does **not** guess correspondence from class
name, semantic role, or geometry. This keeps the workflow object-agnostic and safe for future
custom use cases.

## UI

The Reconstruction sidebar gains **RECONSTRUCTION MEASUREMENTS** with:

- Target
- Source
- Values
- State
- Authority

Only source measurements already carrying an available value are displayed. E2 intentionally
omits uncertainty, reliability/confidence, rejection reasons, calibration/scale validity, and
measurement eligibility details. Those belong to **8.8E3 — Measurement Validity / Confidence
Presentation**.

The authority copy remains explicit:

`Existing measurement evidence • validity/confidence deferred to E3`

E2 cannot claim certified metrology and does not change canonical geometry, semantic, QC, or
reference authority.

## Validation

```bash
./scripts/run_phase8_8e2_tests.sh quick
./scripts/run_phase8_8e2_tests.sh protocol
./scripts/run_phase8_8e2_tests.sh backward
python scripts/run_phase8_8e2_reconstruction_measurement_preview.py
rm -rf /tmp/phase8_8e2_reconstruction_measurement
python scripts/run_phase8_8e2_reconstruction_measurement_smoke.py
cat /tmp/phase8_8e2_reconstruction_measurement/summary.json
python main.py
```

Manual preview checks:

1. The existing D5 semantic selector remains the only object selector.
2. E1 shows the selected object as the inspection target.
3. E2 shows the same target and an explicit measurement source.
4. Values are visible without `±`, reliability, rejection reasons, PASS/FAIL, or reference claims.
5. Changing to another same/different-class object does not transfer evidence automatically.
6. Returning to the originally bound object restores the exact attached evidence.
7. The right sidebar remains vertically scrollable with no horizontal clipping at 1366×768.

No tag is created for E2.
