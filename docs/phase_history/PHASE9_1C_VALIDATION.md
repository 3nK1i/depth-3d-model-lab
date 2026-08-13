# Phase 9.1C Validation — Manual Calibration, Import & Reusable Profiles

Phase 9.1C adds the non-UI management foundation for manual camera calibration,
profile import, and reusable profile persistence. It is additive to the Phase
9.1A profile contract and Phase 9.1B resolution-safe application layer.

## Authority and compatibility rules

- Manual entry creates calibrated camera intrinsics but remains `unvalidated`.
- Native Phase 9 profile imports preserve their original method, validation
  state, timestamp, quality evidence, and provenance.
- Phase 8 live-camera JSON imports remain `legacy_compatible`; they are never
  upgraded to new Phase 9 validation evidence.
- Reusable profiles are stored as deterministic Phase 9 JSON records in an
  explicit filesystem-backed store. QSettings is not used for calibration
  evidence or profiles.
- Store paths are derived from a SHA-256 of the immutable profile ID, preventing
  path traversal through profile names.
- Writes use a temporary file plus atomic `os.replace()`.
- Metric scale, measurement authority, certified metrology, Phase 8 live
  calibration, UI, and external reconstruction runtimes are unchanged.

## Focused validation

```bash
./scripts/run_phase9_1c_tests.sh quick
./scripts/run_phase9_1c_tests.sh integration
python scripts/run_phase9_1c_calibration_smoke.py
git diff --check
```
