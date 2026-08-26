# D4 — Controlled Ten-View Canonical Accumulation

## Scope

D4 expands the qualified D3.1 five-view dense registration chain into a raw ten-view canonical XYZRGB accumulation.

It deliberately does **not** perform voxel fusion, spatial merging, multi-view filtering, persistence, metric-scale qualification, measurement authority, or certified metrology.

## Selection policy

Automatic D4 selection:

1. starts from registered PyCOLMAP workspace order,
2. considers contiguous ten-view windows,
3. requires the supplied qualified five-view chain to remain inside the window,
4. can exclude previously disqualified adjacent dense pairs,
5. ranks the remaining windows by weakest adjacent sparse-track overlap, weakest shared-track count, smallest worst baseline, and total sparse support.

The dense consistency of all nine adjacent links is intentionally **not** assumed from sparse overlap. That becomes the separate D4.1 qualification gate before any fusion work begins.

For the validated office sequence, the intended first D4 experiment preserves the D3.1-qualified `11 → 12 → 13 → 14 → 15` chain while excluding the previously degraded `15 ↔ 16` dense pair. This should force a controlled extension toward earlier views rather than reintroducing known-bad evidence.

## Authority

- SfM gauge remains arbitrary.
- Metric scale is not claimed.
- Raw D4 geometry is validation-only.
- Canonical geometry authority is not promoted.
- Measurement authority is not promoted.
- Gaussian/NeRF authority is unchanged.

## Acceptance for D4

D4 itself passes when:

- all ten selected keyframes are registered and materialized,
- all ten DA3 depth jobs execute through the isolated runtime,
- all ten B1 SfM-depth alignments are accepted,
- all ten C-style canonical projections succeed,
- exactly ten raw clouds are accumulated without merging,
- the source-colored preview shows one plausible common scene region rather than catastrophic coordinate separation,
- the output explicitly records that no fusion/filtering/persistence or metric authority occurred.

D4 does **not** authorize voxel fusion. The next gate must qualify the nine adjacent dense links under unchanged D2-style reprojection-depth consistency thresholds.
