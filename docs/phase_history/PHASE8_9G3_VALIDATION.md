# Phase 8.9G3 — Backward Compatibility & Final Regression Gate

G3 is the final verification substep of Phase 8.9G. It executes the targeted backward-compatibility matrix frozen in G1 after the complete current-tree regression has passed in G2. It is qualification-only: no production source, reconstruction algorithm, dependency, external backend checkout, version metadata, or release tag is changed.

## Strict baseline

The workstation must be on `phase8-reconstruction` at commit `a7cae9f1477a814b7561739a07d4d2cfd1b3f593`, synchronized with `origin/phase8-reconstruction`, with only the five G3 qualification files present as worktree changes.

Release metadata must remain `0.7.5`, and `v0.8.0` must remain absent. The committed G2 validation evidence must be present.

## Frozen G3 compatibility matrix

G3 runs exactly six groups and stops at the first failure:

1. `scripts/run_phase5_tests.sh smoke` — pre-reconstruction core/runtime smoke compatibility.
2. `scripts/run_phase6_step7_tests.sh quick` — Phase 6 model/benchmark release behavior.
3. `scripts/run_phase7_step4_tests.sh quick` — Phase 7 operational-mode release behavior.
4. `scripts/run_phase7_5_step10_tests.sh quick` — Phase 7.5 measurement/inspection release behavior.
5. `scripts/run_phase8_8h3_tests.sh backward` — inherited Phase 8.8 backward-compatibility chain.
6. A bounded actual-PyQt `MainWindow` contract proving the five legacy operational modes remain enabled and route to their existing pages, Reconstruction remains additive, startup still begins on Live 3D, and shutdown closes the application/reconstruction controllers cleanly.

The runner records one log per group plus `/tmp/phase8_9g3_backward_compatibility/summary.json`.

## Deliberate exclusions

G3 does not rerun the complete unit/UI/integration/stress suites; G2 already qualified the full current tree. It does not rerun real PyCOLMAP, DA3, Gaussian, or Nerfacto campaigns and does not install or import their heavy external runtimes into production `depth`.

The bounded `MainWindow` check does not start camera enumeration, inference, benchmark, snapshot, inspection, stable-video processing, or Reconstruction. It validates availability, routing, additive navigation, and clean shutdown only.

## Failure policy

Any failing group blocks G3 closure. Stop at the first failure, diagnose the exact regression, and correct the underlying problem. Do not weaken historical tests, compatibility expectations, authority rules, or isolation requirements simply to make the gate green.

## Closure meaning

A green G3, combined with the already-green G2 complete current-tree regression, closes Phase 8.9G. The next roadmap item is **8.9H — Documentation / Release Preparation**. The application remains version `0.7.5`; the annotated `v0.8.0` release tag is reserved for 8.9I after final approval.
