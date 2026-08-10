# Phase 6 Step 6 — Pillow Deprecation Cleanup

This replacement removes the deprecated explicit `mode="RGB"` argument from the Marigold PIL conversion.

The RGB array is already contiguous `uint8` data with shape `H × W × 3`, so Pillow infers RGB safely through `Image.fromarray(rgb)`.

A focused regression test runs Marigold preprocessing with `DeprecationWarning` promoted to an error.

## Validation

```bash
python -m compileall -q app core tests scripts main.py
pytest -q tests/unit/test_marigold_adapter.py
pytest -q tests/stress/test_marigold_adapter_stress.py
./scripts/run_phase6_step6_tests.sh quick
./scripts/run_phase6_step6_tests.sh full
```
