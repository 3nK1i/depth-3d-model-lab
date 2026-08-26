# D4.2 — Weak-Link Bridge Diagnosis

## Scope

D4.2 is a validation-only diagnostic after a D4.1 ten-view chain has exactly one degraded adjacent dense-depth link.
It does not alter the D4 geometry or choose/remove a keyframe automatically.

For the current D4 chain `6→7→8→9→10→11→12→13→14→15`, D4.1 isolated `7↔8` as the only degraded adjacent pair. D4.2 therefore tests the two skip-one bridges around that weak link:

- `6↔8` (left bridge)
- `7↔9` (right bridge)

Both bridges reuse the exact D4.1/D2 reprojection-depth consistency thresholds recorded in `d4_1_summary.json`.
No PyCOLMAP or DA3 rerun is required because D4 already persisted the per-view canonical cloud, aligned depth provenance, camera pose and camera model evidence needed for the test.

## Interpretation

- **Both bridges pass:** the direct weak pair is pair-specific; neither member is individually disqualified by bridge evidence.
- **6↔8 passes, 7↔9 degrades:** view 7 becomes the preferred *diagnostic* exclusion candidate.
- **6↔8 degrades, 7↔9 passes:** view 8 becomes the preferred *diagnostic* exclusion candidate.
- **Both degrade:** local span instability; a single-view exclusion is not justified yet.
- **Any bridge inconclusive:** collect/select better overlap evidence before inferring a culprit.

Any suggested exclusion remains non-authoritative until a subsequent candidate-chain qualification passes. D4.2 itself never mutates the selected views.

## Authority boundary

The SfM gauge remains arbitrary. D4.2 claims no metric scale, certified metrology, canonical geometry authority or measurement authority. It performs no spatial merge, voxel fusion, multi-view filtering or persistence mutation.
