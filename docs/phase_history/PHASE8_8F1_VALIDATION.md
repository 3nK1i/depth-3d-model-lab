# Phase 8.8F1 — Appearance Representation Controls

## Scope

Phase 8.8F1 adds generic application/UI controls for persisted neural appearance
representations. It consumes existing Phase 8.7F5 persistent-scene artifacts and
keeps Gaussian/NeRF authority strictly `appearance_only`.

F1 does **not** render, train, probe, launch, or switch the Reconstruction
viewport. Gaussian visualization belongs to F2, NeRF visualization to F3,
representation switching to F4, and resource/fallback UX to F5.

## Architecture

- Canonical geometry remains geometry authority.
- Gaussian and NeRF remain appearance-only artifacts.
- F1 reads persisted artifact identity/provider/render-capability/scale context.
- The selector chooses which appearance metadata is inspected; it does not
  activate a renderer.
- No object class or industrial use-case specific logic is introduced.
- No heavy runtime imports are added to the production UI.

## Files

Added:

- `app/reconstruction_appearance_presentation.py`
- `scripts/run_phase8_8f1_appearance_controls_preview.py`
- `scripts/run_phase8_8f1_appearance_controls_smoke.py`
- `scripts/run_phase8_8f1_tests.sh`
- `tests/ui/test_phase8_8f1_appearance_representation_controls.py`
- `tests/unit/test_phase8_8f1_contract.py`
- `tests/unit/test_phase8_8f1_reconstruction_appearance_presentation.py`

Modified:

- `app/widgets/reconstruction_page.py`
- `tests/unit/test_phase8_8e5_contract.py` — narrows the E5 UI source boundary
  to stop at the new F1 group, preventing future sections from being mistaken
  for E5 controls.

No core reconstruction algorithms, workers, backend protocols, Gaussian/NeRF
training code, or Phase 7.5 measurement code are modified.

## Validation

Run:

```bash
./scripts/run_phase8_8f1_tests.sh quick
./scripts/run_phase8_8f1_tests.sh protocol
./scripts/run_phase8_8f1_tests.sh backward
python scripts/run_phase8_8f1_appearance_controls_preview.py
rm -rf /tmp/phase8_8f1_appearance_controls
python scripts/run_phase8_8f1_appearance_controls_smoke.py
cat /tmp/phase8_8f1_appearance_controls/summary.json
python main.py
```

Manual preview checks:

1. `APPEARANCE REPRESENTATIONS` is visible in the Reconstruction settings.
2. The selector exposes persisted Gaussian and NeRF artifacts when present.
3. Switching Gaussian ↔ NeRF changes metadata only; the 3D viewport does not
   switch representation in F1.
4. Provider, artifact, declared render capability, scale context, state and
   appearance-only authority remain readable.
5. A degraded scene with one missing neural artifact remains honest; F1 does
   not fabricate availability.
6. No metric, measurement, inspection/QC, reference, or certified-metrology
   authority is promoted from neural appearance.

Do not commit until focused automated, smoke, visual and normal-app checks pass.
