# Phase 7.5 Step 6 — Import-Isolation Test Correction

## Problem

`test_step6_imports_do_not_load_optional_runtime_frameworks` inspected the shared
pytest process and asserted that `torch`, `PyQt6`, `ultralytics`, and `vispy` were
absent from `sys.modules`.

That assertion was order-dependent. The complete unit suite legitimately imports
PyTorch and Qt in earlier tests, so the test could fail even though importing
`core.measurement.pipeline` did not load those frameworks.

## Correction

The test now launches a fresh Python subprocess, records its initial module set,
imports `core.measurement.pipeline`, and rejects only optional framework modules
that were newly loaded by that import.

No production source file is modified.

## Validation executed

```text
python -m compileall -q app core scripts tests                       passed
pytest -q tests/unit/test_phase75_step6_backward_compatibility.py   3 passed
./scripts/run_phase7_5_step6_tests.sh quick                         19 passed
```

Run the full split-process regression in the configured project environment:

```bash
./scripts/run_phase7_5_step6_tests.sh full
```
