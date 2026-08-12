# Phase 8.8H4 — Phase 8.8 Full Regression

H4 is the broad regression gate for the completed Phase 8.8 application integration work. It adds no production feature or controller/backend behavior.

The authoritative H4 gate recompiles the application/test tree, runs the complete automated `tests/unit`, `tests/ui`, `tests/integration`, and `tests/stress` suites, and then reruns the inherited Phase 8.8 protocol and backward-compatibility chains through H3. The H3 chains recurse through H2/H1 and the completed A–G Phase 8.8 runners.

H4 does not execute real heavy Gaussian/NeRF training/runtime work, does not add device-specific VRAM thresholds, does not claim RTX 5090 profiling, does not alter canonical/semantic/measurement/QC/reference authority, and does not create a release tag. Hardware profiling, endurance qualification on target hardware, real-scene production qualification, release-grade exports/reports, and release closure remain Phase 8.9 work.

Acceptance requires every H4 group to return zero, `summary.json` to report `pass=true`, and exact Git inspection to confirm that H4 changes are limited to H4 validation artifacts plus regression corrections explicitly exposed by the full gate. Any regression failure must be investigated before H5; tests must not be weakened to obtain a pass.

## Full-regression correction note

The first H4 full-unit run exposed two pre-existing regression defects outside the H4 runner itself: the Phase 7 release layout check still referenced two validation documents at their former repository-root locations after those documents had moved under `docs/`, and the Phase 8.8A2 live-controller delegation test used a non-UUID mock session identifier after the G1 session model made UUID session identity authoritative. H4 corrects the release-diagnostic paths and the stale test fixture only; no Reconstruction runtime/controller/backend behavior or authority contract is changed. The complete H4 gate must be rerun from the beginning after these corrections.

The subsequent full run passed the complete unit suite (1703 tests) and complete UI suite (392 tests), then exposed a second historical regression defect during integration collection: Phase 8.8G3 had repurposed `tests/reconstruction_g3_helpers.py` for checkpoint/resume UI fixtures and unintentionally removed the earlier Phase 8.7G3 endurance helper exports still imported by the G3/G4 integration suites. H4 restores those helper exports alongside the newer checkpoint/resume helpers. This is test-surface backward compatibility only; production behavior remains unchanged. The complete H4 gate must again be rerun from the beginning.
