# Phase 9.4F Validation — Gaussian Controls & Scene Layers UX

Phase 9.4F adds professional, presentation-only controls around the already-persisted
9.4E interactive Gaussian appearance. It does not train, render through gsplat,
modify `scene.pt`, change persistence schema, or grant Gaussian geometry/metric/
measurement/QC authority.

## Scope

- Add a **Gaussian appearance** row to the existing Reconstruction `SCENE LAYERS`
  group without changing the legacy 9.3F layer dictionary/API.
- Keep the interaction simple: `Interactive | Preview | Hide` remains the primary
  view-mode switch.
- Add compact interactive display controls:
  - Density: `Full / 75% / 50% / 25%`;
  - Splat size: `60–160%`;
  - Opacity: `25–100%`;
  - `Reset display`.
- Controls deterministically resample/scale only the already-loaded display buffers.
- Canonical surface stays loaded underneath even when visually hidden or covered by
  Gaussian appearance and remains the ruler/picking authority.
- Settings survive hide/show and scene reload within the current app session; they
  are not persisted into the reconstruction scene.

## Authority contract

- Gaussian: `appearance_only`.
- Canonical geometry remains authoritative.
- Gaussian metric-scale, measurement and QC authority remain false.
- `certified_metrology=false`.
- Density/size/opacity are presentation controls only.
- No Gaussian/Graphdeco backend is started and no heavy Gaussian module is imported
  into `depth` by these controls.

## Validation

```bash
./scripts/run_phase9_4f_tests.sh quick
./scripts/run_phase9_4f_tests.sh integration
./scripts/run_phase9_4f_tests.sh backward
```

Real persisted scene, backend-free:

```bash
python scripts/run_phase9_4f_real_gaussian_controls_check.py \
  --scene-manifest /tmp/phase9_4d_persistent_gaussian_scene/persistent_reconstruction_scene.json
```

Manual UI:

1. Load the persisted South Building scene.
2. Open `GAUSSIAN SPLATTING VISUALIZATION` and click `Interactive`.
3. Adjust Density, Splat size and Opacity; the viewport must update immediately.
4. Toggle `Gaussian appearance` from `SCENE LAYERS`; canonical geometry remains
   loaded and ruler picking stays canonical.
5. `Preview` and `Hide` remain mutually coherent with the scene-layer toggle.
6. Reset restores Full / 100% / 100%.

Phase 9.4G remains responsible for performance/resource lifecycle hardening.
