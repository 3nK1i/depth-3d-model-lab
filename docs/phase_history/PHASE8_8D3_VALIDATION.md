# Phase 8.8D3 — Offline Sparse-SfM Health Inspection

## Scope

D3 makes the already-qualified offline sparse-SfM quality gate visible in the Reconstruction UI.
It is deliberately presentation-only:

- no PyCOLMAP worker changes;
- no Stella/native worker changes;
- no new reconstruction thresholds in the Qt layer;
- no dense/canonical geometry;
- no metric-scale promotion;
- no metrology certification.

The UI consumes `sfm_summary.quality_assessment`, which is produced by the existing qualified
SfM backend. It cross-checks reported quality metrics against the result summary when both are
present and degrades to `UNAVAILABLE` rather than inventing a quality classification.

Accepted backend results legitimately report `reasons=[]`. Reconstruction metadata normalization
uses immutable tuples, so an empty JSON array can round-trip through generic metadata as an empty
object. D3 handles that ambiguity **only for the empty `reasons` field**; non-empty mappings remain
invalid. This keeps the presentation schema strict without changing shared metadata serialization.

## UI

The D2 `OFFLINE INSPECTION` section now includes a compact `SPARSE SFM HEALTH` block:

- `Health` — `ACCEPTED`, `DEGRADED`, `REJECTED`, or `UNAVAILABLE`;
- `Registration` — registered/input keyframes plus coverage ratio;
- `Reprojection` — mean reprojection residual in image pixels;
- `Support` — mean views per sparse point plus points per registered camera;
- `Gate` — backend reason summary or `No sparse-SfM gate warnings`.

All wording is operational. The block explicitly states that sparse-SfM health is **not metrology**
and is not a metric-accuracy score.

## Lifecycle

- A completed offline result retains its health presentation with the sparse reconstruction.
- A new offline run clears the previous health block before starting.
- A new live session clears previous offline health/inspection state.
- Optional D1 visualization failure does not destroy a valid backend health result; health remains
  inspectable even when the presentation artifact cannot be loaded.

## Acceptance

Run:

```bash
./scripts/run_phase8_8d3_tests.sh quick
./scripts/run_phase8_8d3_tests.sh protocol
./scripts/run_phase8_8d3_tests.sh backward
```

Expected groups on the D2 baseline:

- quick: `35 passed`;
- protocol: `35`, `2`, `1`, `2`, then `2` passed;
- backward: `21`, `11`, `75`, `64`, `47`, `29`, `19`, `24` passed.

Preview without a backend:

```bash
python scripts/run_phase8_8d3_offline_health_preview.py
```

Real South Building validation:

```bash
rm -rf /tmp/phase8_8d3_offline_health
python scripts/run_phase8_8d3_offline_health_smoke.py \
  --images /home/eiatech/Desktop/phase8_3c_south_building/south-building/images \
  --sfm-python ~/anaconda3/envs/eia-pycolmap-probe/bin/python
```

The real smoke must preserve:

- external PyCOLMAP execution;
- no in-process `pycolmap` import;
- `scene_availability=partial`;
- `offline_scale_status=arbitrary`;
- `canonical_geometry_authoritative=false`;
- `certified_metrology=false`.

No release tag is created for D3.
