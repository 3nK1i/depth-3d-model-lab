# Phase 8.8E1 Validation — Scene / Object Selection

## Scope

Phase 8.8E1 begins the Inspection / Measurement / Reference UI series with one
small application-layer responsibility: bind the **active reconstruction scene**
and the **existing D5 semantic-object selection** into a single generic
inspection-target contract.

E1 deliberately does **not** add another object selector. The D5 `SEMANTIC /
OBJECTS` combo remains the only semantic-object picker. With no semantic object
selected, the active scene is the downstream target; selecting an existing D5
object changes the E1 scope to that object.

E1 does not calculate or evaluate measurements, measurement confidence,
inspection/QC, registration/deviation, or reference comparison. Those remain
8.8E2–E5 responsibilities.

## Architecture findings from the `0f3b13a` baseline

- `ReconstructionPage` already owns the current D5 semantic selection and
  viewport highlighting.
- `ReconstructionController` / `ReconstructionPersistenceController` already
  expose the currently loaded `PersistentReconstructionScene` and scene
  availability.
- `PersistentReconstructionScene` already preserves canonical geometry,
  semantic-geometry, and inspection-QC artifact authority from Phase 8.7F.
- Phase 8.7C inspection/QC and Phase 7.5 Measure & Inspect already exist and are
  not reimplemented by E1.
- The correct smallest E1 boundary is therefore an application/UI selection
  contract plus passive presentation, not a new engine/controller/backend.

## Implemented behavior

- `ReconstructionInspectionSelection` is immutable and object-agnostic.
- Selection scopes are `none`, `scene`, and `object`.
- Persistent scene identity is preferred when present; otherwise the existing
  offline/semantic session provides scene context.
- An existing D5 object selection is promoted only to **selection context**.
- Scene/semantic session mismatch is rejected instead of silently associating
  the wrong object with a scene.
- Canonical geometry authority is preserved only when provided by the loaded
  persistent scene.
- E1 cannot claim certified metrology.
- E1 UI copy explicitly says: `Selection only • measurement/QC not evaluated`.
- The new `INSPECTION TARGET` group adds no new object combo or navigation
  buttons; it reflects the existing D5 selection.
- The page exposes `inspection_selection` and `inspection_target_changed` for
  later E2–E5 integration without re-parsing UI labels.
- Generic test/preview classes use `housing`, `gear`, and `shaft` to ensure the
  production implementation is not connector-specific.

## Files

Modified:

- `app/widgets/reconstruction_page.py`

Added:

- `app/reconstruction_inspection_selection.py`
- `docs/phase_history/PHASE8_8E1_VALIDATION.md`
- `scripts/run_phase8_8e1_scene_object_selection_preview.py`
- `scripts/run_phase8_8e1_scene_object_selection_smoke.py`
- `scripts/run_phase8_8e1_tests.sh`
- `tests/unit/test_phase8_8e1_reconstruction_inspection_selection.py`
- `tests/unit/test_phase8_8e1_contract.py`
- `tests/ui/test_phase8_8e1_scene_object_selection.py`

No `core/reconstruction`, worker protocol, backend environment, model adapter,
measurement engine, QC engine, or reference-registration file is changed.

## Package-preparation validation

Executed in the package-preparation environment:

```text
Baseline ZIP SHA-256:                             PASS
Baseline HEAD/origin/clean report:                PASS
E1 Python compilation:                            PASS
E1 pure unit + authority contract tests:          12 passed
D5 pure semantic/contract + E1 pure regression:   20 passed
Shell syntax:                                     PASS
```

PyQt6 / VisPy are not installed in the package-preparation environment. The
complete E1 quick/UI and real desktop preview/smoke must therefore be run in the
project `depth` environment.

## Local validation

### Focused quick gate

```bash
./scripts/run_phase8_8e1_tests.sh quick
```

Expected target from the packaged test set:

```text
78 passed
```

### Protocol / authority gate

```bash
./scripts/run_phase8_8e1_tests.sh protocol
```

This reruns the D5 protocol boundary and the E1 authority/duplication contract.
No reconstruction protocol is changed by E1.

### Tiny backward sanity

```bash
./scripts/run_phase8_8e1_tests.sh backward
```

This preserves the D-series backward gates and adds only the small existing
Phase 7.5 Measure & Inspect contract/page sanity relevant to avoiding duplicate
selection architecture.

### Interactive preview

```bash
python scripts/run_phase8_8e1_scene_object_selection_preview.py
```

Expected UI behavior:

- one existing D5 semantic-object selector only;
- `INSPECTION TARGET` is visible below `SEMANTIC / OBJECTS`;
- selecting `gear` / `shaft` updates E1 `Scope`, `Object`, `Geometry`, and
  `State` immediately;
- clearing D5 object selection returns E1 to scene scope;
- authority remains `Selection only • measurement/QC not evaluated`;
- no measurement, QC result, or reference comparison appears in E1.

### Automated Qt smoke

```bash
rm -rf /tmp/phase8_8e1_scene_object_selection
python scripts/run_phase8_8e1_scene_object_selection_smoke.py
cat /tmp/phase8_8e1_scene_object_selection/summary.json
```

Required smoke headline:

```text
Phase 8.8E1 scene/object selection smoke: PASS
```

The smoke screenshot is written to:

```text
/tmp/phase8_8e1_scene_object_selection/reconstruction_e1_scene_object_selection.png
```

## Commit gate

Do not commit until focused tests, protocol/backward sanity, actual desktop
preview/smoke, and `git diff --check` are green. No Phase 8 release tag is
created at E1.
