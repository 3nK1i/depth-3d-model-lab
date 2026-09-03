# I5.2B — Current Frozen-Foundation Low-Overlap / Weak-Parallax Replay

## Scope

I5.2B executes the three deterministic Castle-P19 slices frozen by I5.2A through the current production-native Phase 8.3D PyCOLMAP image-sequence path:

1. full-source control;
2. increased-spacing/full-span low-overlap proxy; and
3. contiguous narrow-span weak-parallax proxy.

All three scenarios use the same exhaustive matching policy and the same external PyCOLMAP authority. I5.2B does not start a dense provider, change production sampling, relax SfM/quality thresholds, add a permanent view cap, start MP-SfM, authorize Hybrid, or implement Auto.

## Authoritative stress revalidation

I5.2A source spacing/span are proxies only. I5.2B must validate their actual geometric effect from fresh PyCOLMAP output.

### Low overlap

The decisive quantity is the median number of sparse PyCOLMAP landmarks co-visible in adjacent selected image pairs. The low-overlap stress direction is confirmed only if this authoritative co-visibility count is lower than the full-source control. Normalized common/min-visible overlap and graph connectivity are retained as supporting diagnostics. There is no pass-tuned magnitude threshold.

### Weak parallax

For every sparse landmark observed by at least two registered cameras, I5.2B uses the established I2.4 maximum-supporting-pair formulation:

- maximum pair parallax angle in degrees; and
- maximum camera-baseline / mean-landmark-range ratio.

Both are scale-invariant and therefore comparable across independent arbitrary-gauge SfM reconstructions. Absolute camera baselines are not compared across scenarios. The weak-parallax stress direction is confirmed only when both median parallax and median baseline/range are lower than control.

## Evidence semantics

A real stress replay is evidence even when the requested directional stress effect is not confirmed. Such a result is retained as `DEGRADED`; subsets are not retuned and the replay is not repeated until it passes. Partial registration is preserved as partial registration and never promoted to full-input coverage.

`PASS`, `DEGRADED`, or `REJECTED` here is an I5.2B robustness-evidence classification. It does not select/promote a dense provider and does not claim metric scale, certified metrology, or a surface mesh.

## Runtime boundary

The runner stages exact SHA-bound source bytes, calls `scripts/run_reconstruction_pycolmap_smoke.py --images ... --matching exhaustive` for each frozen slice, persists the real Phase 8.3D reports/logs, re-hashes the staged inputs, and re-verifies the original I5.2A source identity after execution.

A subprocess crash or missing structured Phase 8.3D report is infrastructure failure and is not reclassified as a quality result.

## Next boundary

A structurally complete I5.2B replay authorizes only I5.2C low-overlap / weak-parallax robustness qualification and closure. I5.3 and all later robustness/provider/MP-SfM/Hybrid/Auto work remain deferred to their roadmap boundaries.
