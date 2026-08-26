# H — Dense RGB Reconstruction UI Integration

## Objective

Make the Phase-G persistent canonical **Dense RGB Points** representation a first-class layer in the existing professional Reconstruction workspace without redesigning the page or changing reconstruction authority.

Phase H consumes the self-contained G bundle only. It does not start DA3, PyCOLMAP, fusion, filtering, Gaussian, NeRF, or mesh generation.

## UI integration

The existing `ReconstructionPage` gains a **DENSE RGB POINTS** disclosure group with:

- verified persistent scene identity;
- Extended/Core representation selector;
- point counts;
- qualified source-view count;
- voxel resolution in reconstruction units;
- filtering-confidence/quality summary;
- explicit arbitrary-SfM-gauge status;
- canonical geometry / measurement authority status;
- Load, Fit, and Unload actions.

The existing **SCENE LAYERS** stack gains a `Dense RGB points` visibility switch. This switch changes presentation only and never changes persistence or authority.

The existing `ReconstructionCanvas` renders the verified RGB points through the same VisPy scene/camera architecture used by the Reconstruction workspace. Input XYZ remains in reconstruction coordinates; the existing presentation-only reconstruction→viewer axis conversion is used. Display is deterministically bounded to 250,000 points while the full persistent point count and bounds remain known.

## Representation behavior

- **Extended (primary)** is the default Phase-F/G primary representation.
- **Core (high confidence)** can be selected without re-running reconstruction and without refitting the user's view.
- A new scene load is fully integrity-verified before replacing a currently working Dense RGB presentation.
- A failed/tampered load is fail-closed and preserves the current good scene.
- Unload releases only the Dense RGB viewer layer; the G bundle on disk is unchanged.

## Authority boundary

Phase H runtime presentation enables Dense RGB UI visibility, but the immutable Phase-G manifest remains unchanged.

Required state:

- `canonical_geometry_authoritative = true`
- `canonical_persistence_promoted = true`
- `reconstruction_ui_enabled = true` only in the H presentation layer
- `metric_scale_claimed = false`
- `measurement_authoritative = false`
- `certified_metrology = false`
- `surface_mesh_generated = false`
- no backend started by H

The Phase-G file itself intentionally continues to record `reconstruction_ui_enabled = false` because that field describes the G milestone boundary rather than mutating historical provenance after H loads it.

## Automated validation

`bash scripts/run_h_tests.sh quick`

Expected: **29 passed**.

`bash scripts/run_h_tests.sh affected`

Expected: **86 passed**.

The affected set covers H/G/F contracts plus existing canonical viewer, scene-layer, Gaussian/NeRF layer, unified composition/interaction, workspace UX, and viewport control/layout contracts.

## Manual validation

With a real passing G bundle:

```bash
python scripts/run_h_dense_rgb_ui_preview.py \
  --scene-dir /tmp/eia_g_persistent_dense_rgb_geometry \
  --representation extended \
  --show
```

Check only H-affected behavior:

1. The existing Reconstruction workspace layout remains professional and intact.
2. `DENSE RGB POINTS` appears as a first-class settings group.
3. Extended RGB geometry loads and the viewport fits to it.
4. Source view count, Extended/Core counts, voxel resolution, quality and arbitrary-scale status are readable.
5. Switch to Core; the cloud changes without a backend run and without resetting orbit/pan/zoom.
6. Toggle `Dense RGB points` off/on in Scene Layers; only visibility changes.
7. `Fit` reframes the loaded cloud.
8. `Unload` removes only the Dense RGB layer.
9. Metric/measurement/mesh authority is never presented as qualified.

Full application regression remains the automated-test responsibility; manual checking is limited to the new/affected H interactions plus a small existing-workspace sanity check.

## Next milestone

After automated and real manual UI validation pass:

**I — Offline Dense Reconstruction Quality Campaign**.

## Affected-regression compatibility correction

The first real `affected` run exposed three UI-contract regressions while the H-specific quick suite remained green:

- the new Dense RGB action/combo row established a horizontal minimum that made the 1366×768 settings scroll body 35 px wider than its viewport;
- `dense_rgb_points` was inserted into the legacy Phase-9.3F `_scene_layer_toggles` registry, whose exact key set is a backward-compatibility contract;
- the Dense RGB group was promoted ahead of `RULER METRIC ESTIMATE`, changing the Phase-9.6E4 primary-stack first-four ordering contract.

The correction is presentation-only:

- Dense RGB combo, hint, labels, and action buttons use shrink-safe `QSizePolicy.Ignored` negotiation so the sidebar remains horizontal-scroll-free;
- Dense RGB keeps a dedicated visibility toggle in the same **SCENE LAYERS** UI while the legacy `_scene_layer_toggles` mapping remains unchanged;
- the existing first four primary groups remain `DIGITAL TWIN WORKSPACE`, `BUILD & PROCESSING`, `CANONICAL SCENE`, `RULER METRIC ESTIMATE`, with `DENSE RGB POINTS` immediately after them.

No Dense RGB persistence, authority, geometry, rendering, or backend behavior changes. Expected test counts remain **29 quick / 86 affected**.


## Final Fit framing polish

The explicit Dense RGB **Fit** action now frames only the loaded Dense RGB
representation. It uses conservative 1st/99th percentile presentation bounds so
a small tail of distant retained points, sparse landmarks, cameras, or other
scene context cannot force the main reconstruction to appear unnecessarily
small. The cloud is not cropped, filtered, rewritten, or persisted differently.
Camera azimuth/elevation/FOV are preserved; only center and distance are updated.
