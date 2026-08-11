# Phase 8.8G3 — Checkpoint / Resume UX

G3 exposes the existing whole-engine checkpoint compatibility contract through a bounded Reconstruction-page UX. It does not add a checkpoint format, infer compatibility from metadata alone, launch a resumed engine automatically, or implement G4 failure/recovery behavior.

## Production scope

- `app/reconstruction_checkpoint_resume.py` — dependency-light presentation + exact current resume-context contract.
- `app/reconstruction_controller.py` — stores current plan/provider/payload context, delegates checkpoint loading/compatibility/preparation to the existing persistence controller, and emits preparation/context state.
- `app/widgets/reconstruction_page.py` — compact `Resume` status plus `Load Checkpoint` and `Prepare Resume` controls in the existing `SESSION & PERSISTENCE` group.

A loaded checkpoint is not enough to enable resume. The current `ReconstructionPipelinePlan`, provider identities/versions/execution modes, and rehydrated payload references must be supplied and pass `check_reconstruction_engine_resume_compatibility`. Session identity conflicts also fail closed.

`Prepare Resume` creates the existing `ReconstructionResumePreparation` handoff only. G3 does not automatically start live/offline reconstruction and does not execute heavy Gaussian/NeRF backends.

## Validation

```bash
python -m pytest -q \
  tests/unit/test_phase8_8g3_contract.py \
  tests/unit/test_phase8_8g3_reconstruction_checkpoint_resume.py

python -m pytest -q tests/ui/test_phase8_8g3_checkpoint_resume_ux.py

./scripts/run_phase8_8g3_tests.sh quick
./scripts/run_phase8_8g3_tests.sh protocol
./scripts/run_phase8_8g3_tests.sh backward

rm -rf /tmp/phase8_8g3_checkpoint_resume
python scripts/run_phase8_8g3_checkpoint_resume_smoke.py
cat /tmp/phase8_8g3_checkpoint_resume/summary.json

python scripts/run_phase8_8g3_checkpoint_resume_preview.py
python main.py
```

Expected boundaries: checkpoint manifest/SHA verification remains authoritative; incompatible or missing current context disables `Prepare Resume`; canonical geometry authority is unchanged; metrology is not certified; G4 recovery remains deferred.
