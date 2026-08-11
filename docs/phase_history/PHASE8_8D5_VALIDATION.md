# Phase 8.8D5 Validation — Semantic / Object Presentation

## Scope

Phase 8.8D5 is a presentation-only integration step for the semantic/object geometry already implemented and qualified in Phase 8.7B/F3. It does not rerun perception, alter semantic association, make inspection/QC decisions, calculate measurements, compare against a reference, or change canonical geometry authority.

The Reconstruction workspace can now present a bounded semantic object set, select one object, highlight its semantic support bounds/centroid, focus it, and display class/role/support/quality/scale/authority context.

## Authority rules

- Semantics remain **semantic context** only.
- Inspection/QC retains separate decision authority.
- Semantic confidence is not measurement accuracy.
- Semantic wireframes are support bounds, not dimension boxes.
- Arbitrary geometry remains arbitrary reconstruction gauge.
- Provisional/validated metric semantics never imply certified metrology.
- D5 rejects any semantic payload that claims `certified_metrology=true`.
- D5 does not modify canonical geometry authority.

## Bounded presentation

The UI displays at most 256 semantic objects. The total semantic object count is retained separately so truncation remains explicit. A persistent scene containing only semantic artifact metadata produces an honest summary (`detailed snapshot unavailable`) and never fabricates per-object geometry.

## UI behavior

A new `SEMANTIC / OBJECTS` settings group provides:

- object selector;
- structure summary;
- semantic role;
- support counts;
- semantic/model and vertex-quality context;
- explicit scale semantics;
- explicit semantic authority notice;
- Prev / Focus / Next / Clear controls.

The viewport gains a semantic-context toggle. Normal objects use subdued violet support boxes/centroids; the selected object uses a stronger gold highlight. Starting a new offline or live session clears stale semantic presentation.

The D4 vertically scrollable settings panel and horizontal-shrink contracts are preserved so the additional D5 group does not reintroduce clipping or crushed rows.

## Validation commands

```bash
./scripts/run_phase8_8d5_tests.sh quick
./scripts/run_phase8_8d5_tests.sh protocol
./scripts/run_phase8_8d5_tests.sh backward
```

Preview without an external reconstruction backend:

```bash
python scripts/run_phase8_8d5_semantic_object_preview.py
```

Decisive semantic-engine-to-UI smoke:

```bash
rm -rf /tmp/phase8_8d5_semantic_presentation
python scripts/run_phase8_8d5_semantic_presentation_smoke.py
cat /tmp/phase8_8d5_semantic_presentation/summary.json
```

The decisive smoke first runs the already-qualified Phase 8.7B semantic geometry smoke, then feeds that real semantic result into the D5 Reconstruction presentation and captures a real Qt/VisPy screenshot.

## Manual acceptance

Confirm that:

- the semantic group is readable inside the scrollable settings panel;
- semantic object selection highlights one support volume without hiding the underlying reconstruction context;
- semantic toggle is presentation-only;
- Prev/Next/Focus/Clear work;
- all scale wording remains explicit;
- no D5 control claims defect status, measurement dimensions, metric accuracy, or certified metrology;
- existing D1-D4 offline and C1-C5 live presentation remains intact;
- entering Reconstruction does not autostart a backend.

Do not commit D5 until focused, backward, semantic smoke, and visual/manual gates are green.
