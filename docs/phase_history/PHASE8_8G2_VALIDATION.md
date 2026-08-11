# Phase 8.8G2 — Session Save / Reload Interaction

## Scope

G2 adds explicit user-facing persistent-scene load, save and reload interaction on top of the existing Phase 8 persistence bridge and the G1 reconstruction session model.

It does **not** introduce a second persistence format, export/copy referenced artifact payloads, resume checkpoints, recover failed runtimes, launch reconstruction backends, or change geometry/measurement/inspection/reference authority.

## UX

`SESSION & PERSISTENCE` gains one compact `Actions` status row and three scene actions:

- **Load Scene** — chooses an existing `persistent_reconstruction_scene.json`, validates its manifest checksum and all recorded file references, then binds it through the existing controller.
- **Save Scene** — writes the currently loaded persistent-scene manifest and checksum to a chosen directory. Referenced artifact payloads remain at their recorded paths; this is not an export bundle.
- **Reload** — re-reads the current manifest with integrity checks and requires the same reconstruction session identity.

All three actions are disabled while reconstruction is active. Save/Reload are disabled until a persistent scene is available; Reload additionally requires a known manifest path. Existing manifests in a different target directory require explicit overwrite confirmation.

Checkpoint resume remains deferred to **8.8G3**.

## Safety / authority

- Existing `ReconstructionPersistenceController` read/write functions remain authoritative.
- Reload is same-session only and verifies file references.
- Loading another scene never silently clears or resumes a checkpoint; G1 identity conflict presentation remains visible when evidence belongs to different sessions.
- Canonical geometry remains authoritative.
- G2 never certifies metrology.
- No object-, class-, or industry-specific branching is added.
- No heavy Gaussian/NeRF/SfM/SLAM runtime dependency is added.

## Validation

Run in the production `depth` environment:

```bash
python -m pytest -q \
  tests/unit/test_phase8_8g2_contract.py \
  tests/unit/test_phase8_8g2_reconstruction_session_interaction.py

python -m pytest -q \
  tests/ui/test_phase8_8g2_session_save_reload_interaction.py

./scripts/run_phase8_8g2_tests.sh quick
./scripts/run_phase8_8g2_tests.sh protocol
./scripts/run_phase8_8g2_tests.sh backward

rm -rf /tmp/phase8_8g2_session_save_reload
python scripts/run_phase8_8g2_session_save_reload_smoke.py
cat /tmp/phase8_8g2_session_save_reload/summary.json

python scripts/run_phase8_8g2_session_save_reload_preview.py
```

Manual review should verify compact 1366×768 layout, readable action state/tooltips, no overlap, Load/Save/Reload enablement, no Resume control, and clear distinction between saving a scene manifest versus exporting an artifact bundle.
