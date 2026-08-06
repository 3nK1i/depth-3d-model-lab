# Phase 7.5 Release Test Strategy

## Release objective

Validate that the Phase 7.5 measurement workflow is reproducible, honest about uncertainty, safe on RTX 2060-class hardware, backward compatible with v0.7.0 operational modes, and suitable for a stable release candidate.

## Automated gates

Run each group in a separate process:

```bash
python -m compileall -q app core scripts tests main.py
pytest -q tests/unit
pytest -q tests/ui
pytest -q tests/integration
pytest -q tests/stress
```

The standard optional real Ultralytics and SAM integration tests may remain skipped. Their target-hardware smoke scripts must be run separately.

## New Step 10 gates

```bash
./scripts/run_phase7_5_step10_tests.sh quick
./scripts/run_phase7_5_step10_tests.sh integration
python scripts/run_known_size_validation_smoke.py
```

These validate manifest parsing, export checksums, error metrics, uncertainty coverage, atomic reports, release checks, and backward compatibility.

## Manual validation scope

Only scenarios introduced or affected by Phase 7.5 need full manual validation:

- Real Measure & Inspect analysis
- Object selection
- Mask correction/reset
- Box-method selection
- Measurement and uncertainty presentation
- Temporal stabilization where applicable
- Selected/all-object exports
- HTML/PDF/JSON/CSV/PLY integrity
- Known-size measurement benchmark
- Clean cancellation and shutdown

Backward compatibility requires only a small sanity check of Live 3D, Snapshot, Stable Video, Benchmark, and disabled Reconstruction. The full application regression belongs to automated tests.

## Target hardware

### RTX 2060 6 GB

- Efficient depth/perception profiles
- Sequential isolated model execution
- Maximum 3-5 objects
- Monitor peak VRAM and worker cleanup

### RTX 5090 32 GB

- Auto or higher-quality profiles
- Repeat representative known-size cases
- Confirm no code or configuration change is required

## Release blockers

- Incorrect units or factor-of-1000 errors
- Misleading status colors
- Missing uncertainty disclaimer
- Rejected measurements presented as valid
- Corrupted export manifest
- Automatic mask overwritten by a correction
- Orphaned worker/GPU process
- Model checkpoint tracked in Git
- Known-size thresholds unconfirmed
- Known-size release report failing
- Reconstruction unintentionally enabled

## Release metadata

Version and annotated tag are finalized only after the measured known-size gate and target-hardware release gate pass. Until then the application remains `v0.7.0 - Phase 7 Stable` to preserve truthful release metadata.
