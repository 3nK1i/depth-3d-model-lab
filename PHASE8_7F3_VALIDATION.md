# Phase 8.7F3 — Canonical Geometry + Semantics + QC Integration

## Scope

Phase 8.7F3 binds the already-qualified reconstruction components to the F1
orchestration layer:

- source / pose / depth evidence binding
- dense multi-view fusion + dense quality assessment
- quality-aware canonical surface mesh
- semantic 2D→3D geometry association
- registration / inspection / QC
- provider provenance + artifact authority propagation
- qualified F2 DA3 evidence → canonical geometry bridge

Authority remains explicit:

- source / pose / depth: supporting evidence
- dense geometry + surface mesh: canonical geometry
- semantic geometry: semantic context
- inspection/QC: inspection decision
- DA3 scale: relative/arbitrary unless a separate qualified scale source exists
- `certified_metrology=False`
- reconstruction UI remains disabled

Deferred:

- Gaussian + NeRF appearance integration → 8.7F4
- `PersistentReconstructionScene` + final F lifecycle/provenance/fallback → 8.7F5
- whole-engine hardening → 8.7G

## Focused tests

```bash
./scripts/run_phase8_7f3_tests.sh quick
./scripts/run_phase8_7f3_tests.sh integration
./scripts/run_phase8_7f3_tests.sh stress
```

## Integrated non-UI smoke

```bash
python scripts/run_reconstruction_f3_integration_smoke.py
```

Expected: PASS with completed pipeline, canonical surface authority, 3 semantic
objects, QC pass, no metrology claim, and no UI enablement.

Artifacts are written under:

```text
/tmp/phase8_7f3_integrated/
```

## Real qualified-DA3 → canonical geometry validation

Requires the successful F2 artifacts:

```text
/tmp/phase8_7f2_da3_real/input_manifest.json
/tmp/phase8_7f2_da3_real/worker-output/result_manifest.json
```

Run:

```bash
python scripts/run_da3_canonical_geometry_integration.py
```

Artifacts are written under:

```text
/tmp/phase8_7f3_da3_canonical/
```

The DA3 bridge must preserve:

```text
DA3 = supporting relative/arbitrary evidence
canonical surface = authoritative reconstruction geometry
unit_to_metre = null unless an independent scale source exists
certified_metrology = false
```

Inspect:

```bash
cat /tmp/phase8_7f3_da3_canonical/summary.json
ls -lh /tmp/phase8_7f3_da3_canonical/da3_canonical_geometry_overview.png
google-chrome /tmp/phase8_7f3_da3_canonical/da3_canonical_geometry_overview.png
```

## Backward sanity

Run F2 and F1 quick suites after F3 validation. Earlier NeRF/Gaussian sanity suites
may also be run before commit if desired; F3 contains no neural runtime imports.
