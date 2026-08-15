# Pre-9.4E Interactive Measurement Completion

This completion gate adds one shared two-point distance contract before the interactive Gaussian viewer.

## Product interaction

- **Measure & Inspect:** one ruler icon, then Point A → Point B on the aligned RGB image.
- **Reconstruction:** the same ruler icon, then Point A → Point B on the integrity-verified canonical surface.
- The primary result is deliberately compact: `distance ± uncertainty`.
- Escape/right-click cancels. A third click while the ruler remains active starts a new measurement.

## Measurement truthfulness

- Metric depth can provide an **ESTIMATED** real-world distance before full camera calibration, with explicit heuristic ± uncertainty.
- Calibration/qualification may promote evidence later, but this interaction never claims certified metrology.
- Canonical reconstruction may provide an estimated metric result when an existing provisional `unit_to_metre` scale is bound to the scene.
- An arbitrary-gauge canonical scene with no scale factor shows metric distance as unavailable and may show only reconstruction units. No metre/mm value is invented.
- Gaussian and NeRF remain appearance-only and never become measurement authority.

## Compatibility

Inspection bundle schema remains version 1. Numeric aligned depth, validity, confidence and intrinsics arrays are additive optional NPZ entries. Older bundles remain loadable; their ruler stays disabled until the frame is re-analysed with the new evidence arrays.

The Reconstruction ruler consumes the already-rendered canonical surface and the existing Phase 9.1H/9.3 scale presentation. No persistent-scene schema, reconstruction provider, Gaussian runtime, or NeRF runtime is changed.

## Validation gates

- dependency-light numeric/picking unit tests;
- PyQt Measure & Inspect and Reconstruction ruler tests in the production `depth` environment;
- additive bundle reload integration tests;
- deterministic smoke for estimated metric, provisional canonical metric, arbitrary-scale fail-closed behavior, and projected surface picking;
- narrow backward compatibility over the affected inspection/reconstruction/measurement viewer paths;
- manual ergonomic validation at 1366×768.
