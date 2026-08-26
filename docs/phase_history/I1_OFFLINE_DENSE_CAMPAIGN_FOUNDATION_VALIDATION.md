# I1 — Offline Dense Reconstruction Campaign Foundation

## Scope

I1 adds a validation-only, end-to-end offline campaign orchestrator. It does **not** add a new depth, SfM, fusion, filtering, persistence, UI, metric-scale, measurement, or mesh algorithm.

The runner:

1. executes the existing isolated PyCOLMAP offline SfM path on a video/image sequence;
2. runs the already-qualified isolated DA3 + SfM-depth alignment + canonical projection primitive on registered views;
3. constructs a local dense-consistency graph using the unchanged D2/D4 gates;
4. chooses the longest monotonic **PASS-only** path with bounded skip bridges and no threshold relaxation;
5. fuses that path with the existing `DenseVoxelFusion` kernel;
6. writes an explicit E3-compatible campaign adapter manifest solely so the unchanged formal F and G stages can be reused;
7. executes existing Phase F filtering and Phase G persistence;
8. records stage timings and best-effort host/GPU memory observations;
9. emits `i1_campaign_summary.json`.

A successful I1 run means the generic offline execution/orchestration foundation works. It does **not** mean a scene has passed the Phase I quality campaign; visual/coverage acceptance begins in I2.

## Authority

- SfM scale remains arbitrary.
- No unit-to-metre conversion is introduced.
- Metric scale is not claimed.
- Measurement authority is not promoted.
- Certified metrology is false.
- Mesh generation is not performed.
- G may promote the filtered Dense RGB cloud as canonical geometry/persistence authority only in the same non-metric sense already qualified by G.

## Dense qualification thresholds

I1 reuses the established D2/D4 numerical gates unchanged:

- consistency relative tolerance: `0.12`
- occlusion relative margin: `0.20`
- minimum comparable samples: `1000`
- minimum consistent ratio: `0.50`
- maximum median relative residual: `0.10`
- maximum P90 relative residual: `0.25`
- maximum front-conflict ratio: `0.25`
- spatial NN support scene fraction: `0.05`

Degraded or inconclusive edges are never relaxed to PASS.

## Default campaign bounds

- maximum SfM keyframes: 40
- maximum dense views: 40
- video sampling: 2 FPS
- DA3 process resolution: 392
- maximum bridge gap: 3 workspace orders
- minimum qualified connected path: 10 views
- voxel scene fraction: 0.004
- maximum voxels: 1,500,000
- primary support: at least 2 views
- high support: at least 3 views

## Outputs

```
<output>/
  01_sfm/
  02_dense_views/
  03_pair_diagnostics/
  04_qualified_fusion/
  05_filtering/
  06_persistent_scene/
  logs/
  i1_campaign_summary.json
```

The final G bundle under `06_persistent_scene/` can be loaded directly by the Phase H Dense RGB UI.

## Validation policy

I1 unit tests validate path selection, strict degraded-edge handling, bridge behavior, deterministic dense-view sampling, and schema contracts. Existing F/G/H tests are included in quick/affected campaign gates to guard backward compatibility.

## Real-run correction — pair quality schema / strict JSON

The first complete-video I1 run exposed a schema mismatch in the generic path scorer:
I1 read synthetic `consistent_ratio` / `median_relative_residual` names while the
qualified E3/D2 pair diagnostics author `consistent_ratio_of_comparable` and
`median_absolute_relative_depth_residual`. PASS edges therefore carried fallback
quality `(0, +inf)`. PASS-only connectivity was still respected, but quality
tie-breaking/reporting was invalid and strict `allow_nan=False` JSON publication
correctly rejected the non-finite `+inf` value.

Correction:

- consume the exact E3/D2 pair-quality field names;
- fail closed when a PASS edge lacks finite directed quality evidence;
- keep `+inf` only as an internal empty/single-view path sentinel and serialize it
  as JSON `null`;
- add `--reuse-dense-dir` so a failed I1 campaign can reuse its already-computed
  `02_dense_views` candidates without rerunning DA3. Pair qualification, fusion,
  F filtering, and G persistence are recomputed from the corrected path evidence.

The existing I1 test counts remain unchanged because the synthetic pair fixture was
updated to mirror the real E3/D2 schema: focused 10, quick 45, affected 107 on the
production `depth` environment.
