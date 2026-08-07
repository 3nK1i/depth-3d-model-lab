# Phase 8.4B — Isolated stella_vslam Live Worker

Baseline: `phase8-reconstruction` commit `6a34477836c768e7889a6646b0fbf3cb7db18e60`.

## Scope

Phase 8.4B adds the real native stella_vslam 0.3.13 process boundary while preserving the Phase 8.4A shared-memory and versioned JSON IPC contracts. The production `depth` Python process does not import or dynamically link stella_vslam. Monocular live poses remain arbitrary-scale `T_reconstruction_from_camera` with `unit_to_metre=None`.

Phase 8.4B does not connect the worker to `ApplicationController.frame_captured`; that source integration belongs to 8.4C. Reconstruction UI remains disabled. Loop closure/global optimization are not Phase 8.4 features.

## Target validation

```bash
./scripts/run_phase8_4b_tests.sh quick
./scripts/run_phase8_4b_tests.sh integration
./scripts/run_phase8_4a_tests.sh quick
```

Build the worker against the already-qualified isolated stella installation:

```bash
./scripts/build_stella_live_worker.sh
```

Then run the real process + shared-memory + EuRoC smoke:

```bash
python scripts/run_real_stella_live_worker_smoke.py --max-frames 800
```

Finally run:

```bash
./scripts/run_phase8_4b_tests.sh full
```

Do not commit until all target validation is clean, `git diff --check` is empty, and only the expected Phase 8.4B files are staged.
