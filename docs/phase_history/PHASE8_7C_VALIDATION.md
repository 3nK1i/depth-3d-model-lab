# Phase 8.7C — Registration + Inspection / QC

Phase 8.7C adds a bounded, geometry-first inspection layer on top of the Phase 8.7B persistent semantic 3D objects.

## Scope

- Golden/reference semantic feature contract.
- CAD-derived feature-model source contract (no CAD parser or heavy dependency added here).
- Relative similarity registration for arbitrary-scale reconstructions.
- Metric-aware rigid registration when both sides already carry compatible Phase 8.6B metric scale semantics.
- Explicit engineering tolerances per reference feature.
- Missing, position, orientation and extent deviation findings.
- Connector labels for missing / misaligned / bent pins.
- PASS / WARN / REJECT / INSUFFICIENT_EVIDENCE decisions.
- Geometry quality and uncertainty gating before acceptance.
- JSON reference and inspection artifacts.

## Safety semantics

- Arbitrary-scale comparison remains relative and never emits metric values.
- In arbitrary-scale comparison, the selected registration anchor defines the relative scale; absolute size of that anchor is therefore not a metrology signal. Dimensional QC requires an independent Phase 8.6B metric/reference scale.
- Mixed arbitrary/metric comparison is rejected until an explicit Phase 8.6B scale alignment exists.
- Metric comparison is allowed only when both reference and candidate are already metric.
- Scale uncertainty remains explicit and is not converted into a certified-metrology claim.
- Weak required geometry produces `INSUFFICIENT_EVIDENCE`, never a false PASS.
- `certified_metrology` remains `false`.
- Reconstruction UI remains disabled until Phase 8.8.
- PyCOLMAP and stella_vslam remain external-process isolated.

## Focused validation

```bash
./scripts/run_phase8_7c_tests.sh quick
./scripts/run_phase8_7c_tests.sh integration
./scripts/run_phase8_7c_tests.sh stress
python scripts/run_registration_inspection_smoke.py
```

The deterministic smoke validates four cases:

1. a globally transformed and differently scaled arbitrary reconstruction still passes relative QC after registration;
2. a missing pin plus a bent/misaligned pin produces REJECT with explicit connector defect labels;
3. low-quality required pin geometry produces INSUFFICIENT_EVIDENCE rather than PASS;
4. provisional metric reference/candidate geometry exposes metre deviations while keeping `certified_metrology=false`.

Artifacts are written to:

```text
/tmp/phase8_7c_registration_inspection/reference.json
/tmp/phase8_7c_registration_inspection/pass_report.json
/tmp/phase8_7c_registration_inspection/defect_report.json
/tmp/phase8_7c_registration_inspection/insufficient_evidence_report.json
/tmp/phase8_7c_registration_inspection/summary.json
```

## Full regression

```bash
./scripts/run_phase8_7c_tests.sh full
```

Only the existing optional real-perception skips are acceptable. Phase 8.7C adds no new native worker or UI dependency.
