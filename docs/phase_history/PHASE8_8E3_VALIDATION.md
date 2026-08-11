# Phase 8.8E3 — Measurement Validity / Confidence Presentation

## Scope

E3 qualifies the exact Phase 7.5 measurement observation already bound to the active E1 reconstruction object by E2.

E3 is application/UI presentation only. It reads existing source-authored measurement fields:

- per-kind `available` / `unavailable` / `rejected` status;
- expanded uncertainty and existing lower/upper bounds;
- measurement reliability level/score;
- valid-observation support count;
- uncertainty basis/qualification;
- measurement depth-scale, intrinsics, calibration, and depth confidence-kind context when supplied by the existing inspection bundle;
- warnings and rejection reasons.

E3 does **not** compute measurements, uncertainty, reliability, eligibility, or thresholds. It does not run reconstruction inspection/QC, reference registration/comparison, perception, depth, fitting, PyCOLMAP, Stella, DA3, Gaussian, or NeRF work.

A small E2 compatibility correction is included: Phase 7.5's canonical volume SI unit is `m3`; E2/E3 now both format either `m3` or the legacy display spelling `m^3` as litres.

## Authority boundaries

- E2 values remain existing Phase 7.5 measurement evidence.
- E3 status/reliability/uncertainty remain source-authored Phase 7.5 evidence.
- Semantic/detection confidence is never reused as measurement reliability.
- Metric measurement evidence never promotes an arbitrary reconstruction scene to metric authority.
- E3 cannot claim certified metrology.
- E3 cannot emit inspection/QC PASS/FAIL decisions; E4 owns QC presentation.
- E3 cannot perform reference comparison; E5 owns that workflow.
- E3 production code is object-agnostic.

## Exact provenance rule

E3 requires an already-bound E2 presentation for the active target and rejects a validity payload when either its source or observation ID differs from E2. Changing the D5/E1 target hides both E2 and E3 evidence unless evidence was explicitly bound to that target. Replacing E2 evidence invalidates stale E3 qualification for the same target.

## Validation

```bash
./scripts/run_phase8_8e3_tests.sh quick
./scripts/run_phase8_8e3_tests.sh protocol
./scripts/run_phase8_8e3_tests.sh backward
```

Expected focused results:

- quick: `116 passed`;
- protocol: existing E2 protocol groups plus final E3 contract group `6 passed`;
- backward: existing E2 backward groups plus final affected Phase 7.5 groups `6`, `3`, `1`, and `5` passed.

Interactive UI preview:

```bash
python scripts/run_phase8_8e3_measurement_validity_preview.py
```

Automated Qt smoke:

```bash
rm -rf /tmp/phase8_8e3_measurement_validity
python scripts/run_phase8_8e3_measurement_validity_smoke.py
cat /tmp/phase8_8e3_measurement_validity/summary.json
```

The smoke must report `PASS`, retain an arbitrary reconstruction scene while showing separate metric measurement evidence, present explicit `±` uncertainty plus source reliability/rejection evidence, preserve exact E2 provenance, prove semantic confidence is not reused as measurement reliability, and keep QC/reference/certified-metrology claims false.

## Manual visual gate

At 1366×768, inspect the scrollable Reconstruction sidebar. `SEMANTIC / OBJECTS`, `INSPECTION TARGET`, `RECONSTRUCTION MEASUREMENTS`, and `MEASUREMENT VALIDITY` must remain readable with no overlap or horizontal clipping. The new section must not introduce another object selector or action buttons.
