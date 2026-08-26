# E2 — Consistency-Aware Fusion Refinement Validation

E2 refines the passing E1 ten-view voxel fusion without rerunning PyCOLMAP, DA3, or the D4/D4.3 accumulation.

## Scope

E2 reconstructs E1's exact per-view voxel accounting from the saved D4/D4.3 artifacts and adds evidence that E1 intentionally did not use. The reconstruction must follow the **actual E1 input representation**: E1 loads the saved aligned depth as float32 before projection/fusion. Re-voxelizing the earlier D4 canonical XYZ directly is not equivalent for points lying extremely close to voxel boundaries, because D4/C produced its canonical XYZ from source-model depth promoted to float64 before applying the alignment scale. E2 therefore preserves each saved canonical camera ray, substitutes the float32-quantized aligned depth consumed by E1, and only then reconstructs the per-view voxel observations.

Evidence added by E2 includes:

- robust per-voxel geometric dispersion across independent view centroids;
- 26-neighborhood and radius-2 voxel-lattice connectivity;
- nearby high-support evidence;
- a bounded consistency-confidence score;
- conservative `core` and `extended` multi-view geometry classes;
- an attached single-view fringe artifact kept **diagnostic-only** for coverage inspection.

The primary E2 output is `e2_consistency_extended.npz`; `e2_consistency_core.npz` is the higher-confidence subset. The coverage candidate combines the extended cloud with only immediately attached single-view fringe and is explicitly not promoted to primary geometry.

## Default classification policy

The defaults are deliberately conservative and configurable:

- core, support >= 3: p90 per-view centroid dispersion <= `0.70` voxel edges, sufficiently connected;
- core, support == 2: tighter p90 dispersion <= `0.45` voxel edges plus nearby >=3-view support and local radius-2 density;
- extended: support >= 2, p90 dispersion <= `0.95` voxel edges, connected component >= 64 voxels, local radius-2 density >= 6;
- single-view fringe: retained only as diagnostic coverage when immediately adjacent to at least two extended voxels.

Dispersion is measured in **voxel-edge units**, not metres. E2 preserves the arbitrary offline-SfM gauge.

## Acceptance

E2 passes only when:

- source E1 passed and authorized refinement;
- E1 reported capacity-clean fusion;
- E2 exactly reconstructs every E1 voxel's view-observation count from the ten saved source views using E1-equivalent float32 aligned-depth inputs;
- the core contains at least the configured minimum number of voxels;
- the supported topology contains a sufficiently large connected component;
- no metric, persistence, measurement, mesh, or metrology authority is silently promoted.

## Authority boundary

E2 remains validation-only reconstruction geometry:

- no physical metric scale claim;
- no canonical persistence promotion;
- no measurement authority;
- no certified metrology;
- no surface mesh generation.

Visual validation should compare `extended` RGB, `core` RGB, `extended` confidence/dispersion, and the diagnostic coverage candidate before widening the scene or beginning surface preparation.

## Visual-preview import correction

The E2 preview must import `PointCloudCanvas` from the concrete
`app.widgets.viewport.vispy_canvas` module.  `app.widgets.viewport.__init__`
does not re-export the canvas.  A unit regression check now prevents the
preview from reverting to the invalid package-level import.

### Preview import correction 2

The E2 preview must use the same proven concrete imports as the E1 preview:
`app.widgets.viewport.vispy_canvas.PointCloudCanvas` and
`core.geometry.sampling.enforce_point_budget`.  The non-existent
`core.visualization_sampling` module is not part of this repository.
A source-contract regression test covers both imports.
