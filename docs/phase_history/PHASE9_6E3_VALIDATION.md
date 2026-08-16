# Phase 9.6E3 — NeRF Build / Train / Rebuild UI

Baseline: clean pushed `phase9-digital-twin` at `9fe1448bb3916deeb020b26a264d0abaffb9a801`.

## Purpose

Expose the already-qualified Phase 9.5 canonical-scene → Nerfstudio/Nerfacto → persistent-NeRF chain as one product workflow inside the Phase 9.6 `BUILD & PROCESSING` surface.

## Locked behavior

- Canonical geometry remains the only geometry, picking and measurement authority.
- NeRF remains `appearance_only`; it cannot claim metric scale, measurement, QC, reference or certified-metrology authority.
- `depth` materializes exact canonical RGB/camera evidence only; Nerfstudio/tiny-cuda-nn stay in the external `eia-nerf-nerfstudio` interpreter.
- The external NeRF trainer is the direct E1 process, so E1 process-group cancellation reaches it directly; E3 does not create a detached nested trainer.
- Build creates a new persistent scene revision containing NeRF appearance. Rebuild replaces only the previous NeRF appearance/provider lineage in a new immutable revision; canonical evidence is unchanged.
- The existing Phase 9.5 qualified training bound is retained: Efficient 64, Balanced 128, Max qualified 256 steps. These are bounded build/qualification profiles, not production visual-quality certification.
- Existing NeRF Interactive / Preview / Hide and display/resource controls remain unchanged.

## UI

`BUILD & PROCESSING` remains collapsed by default. With a persistent canonical DA3 scene loaded it exposes:

- NeRF appearance readiness/build state;
- Training profile selector;
- `Build & Train NeRF`, changing to `Rebuild NeRF` when a persisted NeRF already exists;
- the existing shared `Cancel build` action and E1 job progress/provenance.

A successful build is integrity-checked, persisted, loaded through the reconstruction controller, and becomes available to the existing NeRF visualization controls without automatically starting the interactive renderer.

## Deferred

- production-quality long-training policy / benchmarking;
- final visual quality target and global reconstruction optimization;
- generic cross-backend GPU resource coordination (9.6G);
- unified workspace UX consolidation (9.6E4).
## PyQt UI test lifetime correction

The E3 UI regression test retains the `QApplication` Python wrapper for the full `ReconstructionPage` lifetime. This prevents Qt from destroying a newly-created temporary application before the first `QWidget` is constructed on environments where wrapper lifetime is reference-counted aggressively. This is a test-harness correction only; production reconstruction UI code is unchanged.

## Pytest-Qt application lifetime correction

The E3 UI test uses the repository's existing `qtbot` fixture as the sole QApplication owner. It must not create or destroy a separate QApplication because session-scoped Qt objects such as the isolated QSettings fixture remain alive until pytest teardown. This is test-harness-only and does not change production E3 behavior.

## Manual scene-load eligibility correction

Real 1366×768 qualification exposed a scene-bound UI refresh omission: loading a valid
persistent canonical scene updated the canonical viewer but left the E3 NeRF build row
latched in its startup disabled state. `set_persistent_scene()` now refreshes the E3 build
controls on every scene transition. This changes UI eligibility presentation only; NeRF
runtime isolation, persistence, canonical authority and training contracts are unchanged.

