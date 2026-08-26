# I4.1.9A A1.1 — MVS Camera Bridge Stress Closure

## Scope

A1.1 adds targeted fail-closed and determinism coverage around the existing A1
68-view authority-preserving camera bridge. It does not run SfM, bundle
adjustment, image undistortion, PatchMatch Stereo, Stereo Fusion, or any other
heavy MVS operation.

## Added coverage

- missing I4.1.7 workspace fails closed;
- non-empty output evidence is never overwritten;
- non-finite/non-rigid localized poses are rejected;
- duplicate localized landmark observations are rejected;
- references to nonexistent native sparse landmarks are rejected;
- camera-pose authority fingerprints are deterministic and pose-sensitive;
- external worker non-zero exits fail closed;
- external worker timeouts are normalized to `ReconstructionValidationError`;
- a worker that exits successfully without a result JSON is rejected.

`run_i4_1_9a_a1_tests.sh` now supports `focused`, `stress`, and `all` modes.
The no-argument behavior is `all` so A1 unit, integration, and stress coverage
run together by default.

## Authority invariants

A1.1 does not change the A1 camera solution or its authority. The existing
40 global SfM anchors plus 28 accepted localized extra observations remain the
only intended 68-view MVS camera set. Metric scale, authoritative measurement,
certified metrology, and surface mesh authority remain disabled.
