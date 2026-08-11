# Phase 8.8G1 — Reconstruction Session Model

## Scope

G1 establishes one generic application-level reconstruction session composition
for the Session / UX phase. It **does not** replace the existing Phase 8 core
`ReconstructionSession`, persistent-scene manifest, or whole-engine checkpoint
contracts.

The G1 model passively composes already-owned state:

- active/last live or offline session identity;
- selected workflow/source context;
- persistent scene identity and availability;
- checkpoint identity and next-stage metadata;
- generic artifact summaries and authority;
- provider provenance/stages;
- appearance representation kinds;
- canonical-geometry authority and non-certified-metrology status.

A scene/checkpoint/runtime identity mismatch is surfaced explicitly as a session
identity conflict. G1 never silently merges evidence belonging to different
sessions.

## Multi-use-case architecture

G1 is intentionally based on generic concepts only:

`session`, `source`, `workflow`, `scene`, `artifact`, `representation`,
`provider`, `stage`, `checkpoint`, `provenance`, `authority`.

There is no PCB/connector/pin/gear/factory-specific session behavior. Future
industrial use cases and custom providers can contribute artifacts and provider
provenance without changing the session model.

Appearance representation discovery is authority-driven (`appearance_only`) so
future qualified visual representations are not limited to Gaussian/NeRF names.

## UI behavior

The existing compact `SESSION & PERSISTENCE` group gains two passive rows:

- `Context` — focused session origin + normalized lifecycle;
- `Evidence` — scene/checkpoint/artifact/provider/appearance summary.

The existing `Session`, `Scene`, and `Checkpoint` rows remain. G1 adds no save,
open, reload, resume, recover, or recent-session buttons. Those remain owned by
G2-G4; broader layout polish remains G5.

## Boundary

G1 does **not**:

- create a second persistence format;
- save or reload a session/scene (G2);
- execute checkpoint resume (G3);
- implement failure/recovery actions (G4);
- redesign the Reconstruction UI (G5);
- run reconstruction/neural backends;
- modify canonical geometry, measurement, inspection/QC, or reference authority;
- claim certified metrology.

## Files

Added:

- `app/reconstruction_session_model.py`
- `docs/phase_history/PHASE8_8G1_VALIDATION.md`
- `scripts/run_phase8_8g1_session_model_preview.py`
- `scripts/run_phase8_8g1_session_model_smoke.py`
- `scripts/run_phase8_8g1_tests.sh`
- `tests/ui/test_phase8_8g1_reconstruction_session_model.py`
- `tests/unit/test_phase8_8g1_contract.py`
- `tests/unit/test_phase8_8g1_reconstruction_session_model.py`

Modified:

- `app/reconstruction_controller.py` — exposes a computed G1 model + change signal;
- `app/widgets/reconstruction_page.py` — passive compact presentation only.

## Validation

```bash
./scripts/run_phase8_8g1_tests.sh quick
./scripts/run_phase8_8g1_tests.sh protocol
./scripts/run_phase8_8g1_tests.sh backward
```

### Decisive smoke

```bash
rm -rf /tmp/phase8_8g1_session_model
python scripts/run_phase8_8g1_session_model_smoke.py
cat /tmp/phase8_8g1_session_model/summary.json
```

The smoke proves consistent scene/checkpoint composition, explicit identity
conflict detection, conflict recovery, generic future appearance representation
support, authority preservation, and no G2/G3/G4 action execution.

### Interactive preview

```bash
python scripts/run_phase8_8g1_session_model_preview.py
```

Manual checks:

1. `SESSION & PERSISTENCE` remains compact and scrollable at 1366×768.
2. `Session` shows the focused UUID.
3. `Context` shows `Persistent scene • Ready` for the demo.
4. `Evidence` summarizes scene/checkpoint/artifacts/providers/appearance without
   domain-specific terminology.
5. Scene/checkpoint manifest names remain readable.
6. No save/reload/resume/recovery controls are introduced in G1.
