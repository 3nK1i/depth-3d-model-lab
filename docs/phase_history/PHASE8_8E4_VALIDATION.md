# Phase 8.8E4 — Inspection / QC Presentation

## Scope

Phase 8.8E4 adds a read-only Reconstruction UI bridge for the existing Phase 8.7C inspection report.

E4 may present only source-authored inspection/QC evidence:

- overall scene decision;
- exact selected-object decision when `candidate_object_id` matches the active E1 object ID;
- source findings;
- report coverage/statistics;
- registration status/confidence already present in the report;
- relative-only versus metric-comparison availability;
- explicit inspection-decision authority and non-certified-metrology status.

E4 does **not** run registration, apply tolerances, compute deviations, derive PASS/WARN/REJECT from semantic or measurement evidence, or load/select references. The interactive reference-comparison workflow remains Phase 8.8E5.

## Object-agnostic rule

Production E4 code binds QC only by exact session ID and exact `candidate_object_id`. It does not branch on application classes or domain defect names. Domain-specific examples may remain in existing Phase 8.7C validation fixtures, but the E4 application layer stays generic.

## Files

Added:

- `app/reconstruction_inspection_qc_presentation.py`
- `docs/phase_history/PHASE8_8E4_VALIDATION.md`
- `scripts/run_phase8_8e4_inspection_qc_preview.py`
- `scripts/run_phase8_8e4_inspection_qc_smoke.py`
- `scripts/run_phase8_8e4_tests.sh`
- `tests/ui/test_phase8_8e4_inspection_qc_presentation.py`
- `tests/unit/test_phase8_8e4_contract.py`
- `tests/unit/test_phase8_8e4_reconstruction_inspection_qc_presentation.py`

Modified:

- `app/widgets/reconstruction_page.py`

No core reconstruction, worker, protocol, measurement, perception, registration/QC algorithm, or backend files are modified.

## Validation

Run:

```bash
./scripts/run_phase8_8e4_tests.sh quick
./scripts/run_phase8_8e4_tests.sh protocol
./scripts/run_phase8_8e4_tests.sh backward
```

Interactive generic preview:

```bash
python scripts/run_phase8_8e4_inspection_qc_preview.py
```

Decisive existing-engine-to-UI smoke:

```bash
rm -rf /tmp/phase8_8e4_inspection_qc
python scripts/run_phase8_8e4_inspection_qc_smoke.py
cat /tmp/phase8_8e4_inspection_qc/summary.json
```

The smoke runs the existing Phase 8.7C engine first, then passes its `InspectionReport` into the E4 presentation layer. E4 itself does not invoke the QC engine.

## Manual gate

In the preview, confirm:

1. `SEMANTIC / OBJECTS`, `INSPECTION TARGET`, and `INSPECTION / QC` remain readable in the scrollable settings sidebar.
2. The selected generic object receives only its exact source-authored decision/findings.
3. Selecting an object not represented by a feature result shows `NOT EVALUATED • Object`; it must not inherit the scene decision.
4. Clearing object selection shows the source scene decision only.
5. `Authority` explicitly states that the source report owns the inspection decision, metrology is not certified, and reference workflow is deferred to E5.
6. No reference picker, tolerance editor, recompute action, or duplicate object selector is introduced.

Do not commit until automated, smoke, visual, normal-app, `git diff --check`, and changed-file-scope gates are green.
