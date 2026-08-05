# Phase 7 Release Test Strategy

## Release target

`v0.7.0 — Phase 7 Stable`

The release preserves the complete Phase 6 multi-model runtime and validates the four operational pages delivered through Phase 7: Live 3D, Benchmark, Snapshot and Stable Video.

## Automated gates

### Static and metadata

- Python 3.11/3.12 compatibility
- Required project layout
- Version consistency across `core/version.py`, `pyproject.toml` and `config/app.yaml`
- Base and model dependency availability
- Registered checkpoint/source availability
- Writable log, release, benchmark, snapshot and Stable Video directories

### Unit

- Registry and adapter contracts
- Benchmark engine and reports
- Snapshot CLI, atomic export, point/color/mask alignment and manifest semantics
- Stable Video atomic export, resolution adaptation, CSV and manifest generation
- Release artifact validators and atomic release reports

### UI

- Live 3D, Stable Video, Snapshot and Benchmark enabled
- Reconstruction remains disabled
- Repeated mode navigation
- Benchmark responsiveness/cancellation
- Snapshot capture/run/cancellation
- Stable Video lifecycle, statistics and controls

### Integration and stress

- Inference lifecycle and recovery
- Repeated model adapter load/infer/unload
- Benchmark process isolation
- Snapshot and Stable Video repeated artifact validation
- Stable Video sustained frame export/backpressure
- Worker and Qt test groups run in isolated pytest processes

## Release artifacts required

The final gate requires:

- Current-version cross-model benchmark report with all production models passing
- One complete Snapshot export
- One complete Stable Video export
- Full automated suite
- Physical CUDA probe
- RTX 2060, RTX 5090 and RTX 5090 pressure-profile simulations
- Clean Git tree after the release commit

## Release rule

No tag is created until the full release gate passes with all required artifacts and a clean Git worktree.
