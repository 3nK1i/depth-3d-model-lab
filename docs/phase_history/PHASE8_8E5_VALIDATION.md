# Phase 8.8E5 — Reference Comparison Workflow

## Scope

Phase 8.8E5 closes the Phase 8.8E series by making the reference identity and comparison evidence explicit in the Reconstruction workspace.

E5 is application/UI integration only. It consumes:

- the existing E1 scene/object target;
- an explicit Phase 8.7C `ReferenceAssembly` JSON artifact (or equivalent `as_dict()` payload);
- the existing E4-attached Phase 8.7C `InspectionReport` for the same reconstruction session.

E5 does **not**:

- parse CAD meshes;
- build a reference assembly;
- run registration;
- apply tolerance rules;
- compute deviations;
- compute or change QC decisions;
- infer comparison from class name, semantic role, segmentation confidence, or measurement confidence;
- convert arbitrary reconstruction gauge to metric units;
- claim certified metrology.

## Workflow

The `REFERENCE COMPARISON` section provides:

- Target — active E1 scene/object target;
- Reference — explicit reference ID + source;
- Structure — feature/anchor counts and scene profile;
- Registration — source-authored 8.7C registration status, confidence, method and residual;
- Deviation — source-authored quantitative deviation for the exact selected object, or registration residual for scene scope;
- Evidence — exact reference feature identity, geometry quality and uncertainty when available;
- Scale — explicit relative-only versus metric-compatible comparison semantics;
- State — reference/report readiness and mismatch handling;
- Authority — explicit statement that E5 does not register/apply tolerances and is not certified metrology.

`Load JSON` accepts the JSON emitted by Phase 8.7C `write_reference_assembly_json`. `Clear` removes only the active reference binding; it does not delete the existing E4 inspection report.

## Safety gates

1. Reference IDs must match exactly before an E4 report can be used for E5 comparison.
2. Reconstruction session IDs must match exactly.
3. Object deviations are bound only by exact `candidate_object_id`.
4. Switching to another reference ID suppresses the old comparison instead of reusing stale evidence.
5. An object with no exact source result reports no object deviation; scene comparison is not copied to it.
6. Metric values appear only when the source 8.7C report says `comparison_metric_available=true`.
7. `certified_metrology` remains false.
8. E5 production code contains no connector/pin/class-specific branch.

## Focused validation

```bash
./scripts/run_phase8_8e5_tests.sh quick
./scripts/run_phase8_8e5_tests.sh protocol
./scripts/run_phase8_8e5_tests.sh backward
```

Interactive generic preview:

```bash
python scripts/run_phase8_8e5_reference_comparison_preview.py
```

The preview writes its reference artifact to:

```text
/tmp/phase8_8e5_reference_preview/reference.json
```

Decisive existing-engine smoke:

```bash
rm -rf /tmp/phase8_8e5_reference_comparison
python scripts/run_phase8_8e5_reference_comparison_smoke.py
cat /tmp/phase8_8e5_reference_comparison/summary.json
```

The decisive smoke runs the existing Phase 8.7C reference/registration/QC engine first, writes the existing ReferenceAssembly JSON artifact, then hands those artifacts to E5. E5 itself performs no registration, deviation or tolerance computation.

Expected headline:

```text
Phase 8.8E5 reference comparison smoke: PASS
```

Final normal-application sanity:

```bash
python main.py
```

Close normally and require exit code 0.

## Phase boundary

Successful E5 validation closes:

- E1 Scene / Object Selection
- E2 Reconstruction Measurement UI
- E3 Measurement Validity / Confidence Presentation
- E4 Inspection / QC Presentation
- E5 Reference Comparison Workflow

The next roadmap letter is Phase 8.8F — Neural Visualization.
