# Phase 8.8A1 — Dormant Reconstruction Controller Foundation

## Scope

8.8A1 introduces the application-level reconstruction coordinator only. It does not enable the Reconstruction mode, add a Reconstruction page, or start any reconstruction backend.

The controller:

- owns the existing `LiveSlamCaptureController` from Phase 8.4;
- preserves `ApplicationController` as the single live capture owner;
- forwards the existing `LiveCaptureRuntimeState`, live observation, and error contracts without defining a second reconstruction state machine;
- remains dormant on construction;
- performs deterministic, idempotent shutdown;
- keeps a `MainWindow._live_slam_controller` compatibility alias to the same owned bridge.

## Required baseline

- branch: `phase8-reconstruction`
- HEAD: `bb3ca753b3535100503b562c403a7439d1b53803`
- `origin/phase8-reconstruction` at the same commit
- clean working tree before overlay

## Explicitly not part of A1

- no Reconstruction tab enablement;
- no Reconstruction page or mode routing;
- no Stella start/pause/resume UI lifecycle yet;
- no offline reconstruction worker;
- no PyCOLMAP, Gaussian, NeRF, or DA3 startup;
- no reconstruction algorithm/core changes;
- no release metadata or tag changes.

## Validation

```bash
./scripts/run_phase8_8a1_tests.sh quick
```

The focused gate covers:

- dormant controller construction;
- authoritative live-state passthrough;
- signal forwarding;
- idempotent shutdown;
- absence of heavy/native backend imports or startup calls;
- `MainWindow` ownership wiring;
- Reconstruction still disabled and absent from the page stack;
- Phase 8.4 bridge backward compatibility;
- Phase 8.7G4 UI-readiness lock sanity.

No real backend or visual validation is required in A1 because the new controller is deliberately dormant and makes no UI change. Real Stella lifecycle validation begins in 8.8A2.
