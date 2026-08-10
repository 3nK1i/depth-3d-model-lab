# Phase 8.4A Validation — Live SLAM Foundation

Phase 8.4A is a dependency-light foundation step. It does **not** integrate the
native stella_vslam worker yet and it does **not** enable Reconstruction mode.

## Scope

Phase 8.4A adds:

- explicit arbitrary-scale live SLAM pose contracts using
  `T_reconstruction_from_camera`;
- calibrated-camera requirements for production live SLAM requests;
- USB / RTSP live-source support plus recorded-video simulated-live support;
- tracking states and quality/map snapshot contracts;
- a bounded 3-slot POSIX shared-memory frame ring with overwrite detection;
- a latest-frame-only mailbox so control IPC cannot accumulate stale frame work;
- an adapter from the existing stable `SourceFrame` contract into the live SLAM
  frame envelope without changing `SourceFrame`, with explicit capture-clock
  timestamp semantics;
- versioned JSON IPC message contracts for the future native worker;
- native worker endpoint/lifecycle contracts for Unix-domain-socket execution;
- a deterministic mock live SLAM backend;
- focused unit, integration, stress, and non-UI smoke validation.

## External backend qualification already completed

Before Phase 8.4A code was generated, stella_vslam was qualified outside the
repository using an isolated native build:

- stella_vslam `0.3.13`, commit
  `7902892d445843563d2d3149505fd3e1a118a141`;
- stella FBoW submodule commit
  `7753c2c17dd7ce8bdf4a7a3fb1262b78d5b9b9b5`;
- EuRoC `MH_01_easy` monocular tracking completed with return code `0`;
- 3681 frames were reported in the trajectory dump and 135 keyframes in the
  final validation run;
- median tracking time was about 16.6 ms and mean about 18.2 ms;
- this probe remained outside the repository and did not modify `depth`.

Phase 8.4A records only backend-neutral contracts; the native stella worker is
still deferred to Phase 8.4B.

## Deliberately deferred

- native stella_vslam worker executable integration;
- application-controller / USB / RTSP wiring;
- loop closure and global optimization (Phase 8.5);
- metric scale recovery / dense depth fusion (Phase 8.6);
- Reconstruction UI activation (Phase 8.8).

## Safety invariants

- `v0.7.5` release identity remains unchanged.
- Reconstruction mode remains disabled.
- `SourceFrame(image, frame_id, timestamp)` remains unchanged.
- main `depth` remains PyCOLMAP-free.
- offline PyCOLMAP architecture remains untouched.
- native stella_vslam is not imported or linked into `core/reconstruction`.
- full live frame pixels never belong in JSON IPC messages.
- monocular live poses remain arbitrary-scale:
  `unit_to_metre=None`.
- Phase 8.4A rejects loop/global optimization enablement.

## Validation

From the repository root in the `depth` environment:

```bash
./scripts/run_phase8_4a_tests.sh quick
./scripts/run_phase8_4a_tests.sh integration
./scripts/run_phase8_4a_tests.sh stress
python scripts/run_live_slam_foundation_smoke.py
```

Then run the previous reconstruction gate:

```bash
./scripts/run_phase8_3d_tests.sh quick
```

For the final candidate, run:

```bash
./scripts/run_phase8_4a_tests.sh full
```

Expected target totals, derived from the validated Phase 8.3D baseline plus the
new additive Phase 8.4A tests, are:

- 661 unit passed;
- 185 UI passed;
- 24 integration passed, with the same 2 optional perception hardware skips;
- 40 stress passed.

The two optional perception hardware tests may remain skipped exactly as in the
validated v0.7.5 / Phase 8.3 baseline. No Phase 8.4A commit should be made until
the target machine completes the required validation.
