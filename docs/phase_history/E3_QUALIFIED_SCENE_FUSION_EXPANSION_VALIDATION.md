# E3 — Qualified Scene Fusion Expansion

## Purpose

E3 expands the validated E2 ten-view seed into a materially larger offline dense reconstruction while preserving the qualification discipline established in D2/D4.  New views are not admitted by temporal proximity alone.

The trusted E2 seed remains unchanged:

`5 -> 6 -> 7 -> 9 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15`

Automatic E3 candidate discovery is restricted to registered SfM views **outside** the seed extent.  The intentionally excluded interior order 8 is therefore not silently reopened.

Default candidate pool on the current 40-view session:

- left: nearest five registered views before order 5 (expected orders 0..4),
- right: nearest eight registered views after order 15 (expected orders 16..23).

Every new candidate receives:

1. isolated DA3 depth inference,
2. SfM↔dense-depth scale alignment,
3. canonical RGB-D projection in the existing arbitrary SfM gauge,
4. D2-style bidirectional reprojection/depth-consistency qualification.

E3 reuses the exact D4.3/D4.1/D2 thresholds.  No threshold relaxation is allowed.

## Bridge policy

The left and right expansion sides are solved independently as monotonic pass-only paths.  By default an edge may bridge at most three registered workspace orders.  This is deliberately able to bypass a bad direct transition such as the already-known 15↔16 link without declaring order 16 or its neighbours globally invalid.

Path selection priority is:

1. most newly qualified views,
2. largest workspace-order span,
3. strongest worst-case bidirectional consistency,
4. lowest worst median relative depth residual.

Degraded and inconclusive edges cannot enter the selected path.

## Fusion policy

The selected expanded chain is fused with the existing `DenseVoxelFusion` production kernel.  E3 reuses the exact E1 voxel size rather than retuning resolution during coverage expansion.

E3 outputs:

- all fused voxels,
- primary multi-view-supported voxels (default support ≥ 2 views),
- high-support voxels (default support ≥ 3 views).

The default capacity increases to 1,500,000 voxels to accommodate the larger qualified view set.  Any capacity drop fails the E3 gate.

## Qualification gate

E3 passes only if all of the following hold:

- at least 8 new views are selected by qualified growth paths,
- at least 2 new views are added on each side of the ten-view seed,
- every selected expansion edge has status `pass`,
- all selected frames are integrated,
- voxel capacity is clean (`capacity_drops == 0`),
- at least 1,000 supported voxels remain,
- supported-voxel count does not regress below E1,
- the robust supported-cloud spatial span does not regress materially below E1.

A passing E3 result authorizes **F — Confidence / Consistency / Artifact Filtering**.

## Authority boundaries

E3 does **not**:

- establish physical metric scale,
- promote canonical geometry authority,
- promote measurement authority,
- persist canonical dense geometry (G),
- integrate the Reconstruction UI (H),
- formalize confidence/artifact filtering (F),
- generate a surface mesh (J),
- certify metrology.

The SfM gauge remains arbitrary.

## Validation

Packaged implementation validation:

- Python compile: PASS
- focused E3 unit tests: 11 passed
- E3 quick suite: 41 passed
- E3 affected suite: 164 passed
- preview imports use the proven concrete modules:
  - `app.widgets.viewport.vispy_canvas.PointCloudCanvas`
  - `core.geometry.sampling.enforce_point_budget`
- package scope: five new files, zero existing-file modifications

Real-scene validation remains target-hardware work because it requires the user's isolated DA3 CUDA environment and existing `/tmp` E2/E1/D4/B2 artifacts.
