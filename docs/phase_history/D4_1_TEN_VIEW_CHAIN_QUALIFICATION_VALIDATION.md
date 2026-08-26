# D4.1 — Ten-View Dense Chain Qualification

## Purpose

D4.1 is the final raw multi-view registration gate before voxel fusion. It consumes a successful **unfused D4 ten-view accumulation** and applies the existing D2 bidirectional reprojection/depth-consistency diagnostic to every adjacent pair.

Required links:

`0↔1, 1↔2, 2↔3, 3↔4, 4↔5, 5↔6, 6↔7, 7↔8, 8↔9`

The ten-view chain passes only when **all nine** adjacent pairs pass in both directions under the unchanged D2 thresholds.

## Non-goals / authority boundary

D4.1 does not rerun PyCOLMAP or DA3. It does not fuse, filter, persist, smooth, rescale, or modify the D4 cloud. It does not establish metric scale, geometry authority, measurement authority, or certified metrology. The SfM gauge remains arbitrary.

## Default qualification thresholds

- relative consistency tolerance: `0.12`
- occlusion margin: `0.20`
- minimum comparable samples: `1000`
- minimum consistent ratio: `0.50`
- maximum median relative residual: `0.10`
- maximum P90 relative residual: `0.25`
- maximum front-conflict ratio: `0.25`

These are reconstruction-consistency gates, not metric-accuracy specifications.

## Validation

```bash
bash scripts/run_d4_1_tests.sh quick
bash scripts/run_d4_1_tests.sh affected
```

Real D4 qualification:

```bash
rm -rf /tmp/eia_d4_1_ten_view_qualification

python scripts/run_d4_1_ten_view_chain_diagnostics.py \
  --d4-dir /tmp/eia_d4_ten_view_accumulation \
  --output-dir /tmp/eia_d4_1_ten_view_qualification
```

The command returns `0` for PASS, `2` for DEGRADED, and `3` for INCONCLUSIVE while still writing the diagnostic evidence.

Preview any adjacent pair, for example the first new D4 link:

```bash
python scripts/run_d4_1_ten_view_chain_preview.py \
  --d4-1-dir /tmp/eia_d4_1_ten_view_qualification \
  --pair 0-1 \
  --color-mode residual \
  --overlay-sparse \
  --show
```

Residual colors preserve the existing D2 convention: green = consistent, red = front conflict, amber = moderate back-surface disagreement, gray = occluded, white = sparse SfM landmarks.

## Acceptance gate

D4.1 is green only when the summary records:

- `phase = "D4.1"`
- `pass = true`
- `status = "pass"`
- `adjacent_pair_count = 9`
- every adjacent pair has `status = "pass"`
- `spatial_merge_applied = false`
- `voxel_fusion_applied = false`
- `multi_view_filtering_applied = false`
- `metric_scale_claimed = false`

Only then is the ten-view registration foundation qualified for the next milestone: real multi-view voxel fusion.
