# Phase 8.8C3 — Live Sparse Map / Landmark Visualization

## Scope

C3 extends the already-validated C1/C2 Reconstruction viewport with the first
real live sparse scene geometry. It also refines the true world-origin XYZ axes
with restrained 3-D arrowheads matching the visual language of the existing
Live 3D renderer.

C3 does **not** add dense reconstruction, DA3 geometry, a mesh, Gaussian
Splatting, NeRF, metric scale, or certified metrology.

## Data boundary

The isolated native Stella worker already owns Stella's map publisher. C3 adds
an optional, bounded sparse landmark sample inside the existing protocol-v1
`map_snapshot` payload:

- coordinates are Stella reconstruction/map coordinates;
- maximum native sample size: 4,096 landmarks;
- samples are emitted when the map revision changes and periodically every 10
  submitted frames so local optimization can refresh presentation geometry;
- no camera/video capture ownership moves into the worker;
- the production Python process still does not import or link `stella_vslam`.

`LiveSlamObservation` and the base `LiveMapSnapshot` dataclass shapes remain
unchanged. A backward-compatible `LiveMapGeometrySnapshot` subclass carries the
optional coordinates when present. Older protocol-v1 observations without the
new optional field continue to parse as `LiveMapSnapshot`.

## Viewer behavior

- sparse landmarks are transformed only by the established reconstruction to
  viewer Z-up presentation transform;
- the sparse map is replaced by the latest worker sample rather than accumulated
  as duplicate map points;
- display is bounded to 8,192 points even if a future producer supplies more;
- sparse-map visibility is a presentation-only toggle;
- map geometry is retained after clean Stop and cleared only when a genuinely
  new live session starts;
- the existing cyan camera trajectory and red current-camera frustum remain
  visually distinct;
- Fit includes both camera trajectory and sparse-map extent;
- the world axes keep adaptive C2 sizing and now have true scene-space
  arrowheads at X/Y/Z tips.

## Gauge / safety

Live monocular Stella geometry remains arbitrary gauge:

- `scale_status = arbitrary`;
- `unit_to_metre = null`;
- no world alignment is claimed;
- no measurements or certified metrology are enabled by C3.

The bottom-left `VIEWER Z-UP • ARBITRARY GAUGE` badge remains authoritative.

## Validation gates

```bash
./scripts/run_phase8_8c3_tests.sh quick
./scripts/run_phase8_8c3_tests.sh protocol
./scripts/run_phase8_8c3_tests.sh backward
```

Because C3 changes the native Stella worker source, rebuild the isolated worker
before the real runtime smoke:

```bash
./scripts/build_stella_live_worker.sh
```

Then run the real visual smoke from a normal desktop/display session:

```bash
python scripts/run_phase8_8c3_live_sparse_map_smoke.py \
  --max-frames 500 \
  --playback-speed 2.0
```

The smoke must show real sparse points, live trajectory, current-camera frustum,
world-axis arrowheads, arbitrary-gauge safety, clean Stop retention, and clean
worker disposal.
