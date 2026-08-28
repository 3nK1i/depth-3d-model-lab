# I4.3G2A.3-A1 — Low-texture construct-validation amendment

## Why this amendment exists

The original G2A.3 low-texture gate compared the ratio of a log-compressed
aggregate texture proxy against a fixed `0.75` threshold. The proxy itself is:

`log1p(laplacian_variance) + log1p(contrast_stddev) + 3 * edge_density`.

Pre-provider construct validation demonstrated that this ratio is not a valid
percentage measure of texture reduction. Known low-texture positive controls
showed large reductions in raw Laplacian variance, contrast, and edge density
while the log-compressed aggregate ratio remained near `0.90`. The logarithmic
terms intentionally compress dynamic range, so applying a percentage-style
ratio threshold to their sum can reject genuine low-texture data.

No SfM, MVS, DA3, Auto, hybrid execution, or provider-quality result was
consumed before this problem was identified. The correction therefore occurs
before comparative G2B results and does not tune the benchmark to favor either
provider.

## Frozen A1 replacement gate

The original `0.75` material-reduction criterion is retained, but it is applied
to raw component ratios where percentage comparisons are meaningful.

For the complete-room reference and candidate, compute medians of:

- Laplacian variance;
- grayscale contrast standard deviation;
- Canny edge density.

Then compute candidate/reference ratios for each component. A candidate is
objectively low-texture only when:

`edge_density_ratio <= 0.75`

and at least one independent supporting component also satisfies:

`laplacian_variance_ratio <= 0.75` **or** `contrast_stddev_ratio <= 0.75`.

For deterministic contiguous-subset selection, define the qualification margin
as:

`max(edge_density_ratio, min(laplacian_variance_ratio, contrast_stddev_ratio))`.

The lowest-margin contiguous window is selected. It qualifies exactly when the
margin is `<= 0.75`.

## Legacy proxy status

The original log-compressed aggregate proxy remains recorded for diagnostics
and historical comparison only. It has **no qualification authority** after
A1. Evidence explicitly records this fact so a hand-edited snapshot cannot
silently restore the retired gate.

## Provenance and fail-closed behavior

A1 bumps the G2A.3 snapshot schema and freezes an amendment record containing:

- amendment ID `I4.3G2A.3-A1`;
- predecessor and replacement methods;
- unchanged `0.75` criterion;
- exact logical qualification rule;
- explicit false execution/provider-consumption flags;
- explicit retirement of the legacy aggregate ratio for qualification.

Final G2A rehashes bound files/contact sheets and recomputes the raw component
ratios. Policy drift, amendment drift, source mutation, diagnostic mutation,
or reactivation of the legacy proxy fails closed.

## Authority locks

A1 does not change reconstruction authority. PyCOLMAP remains the sole global
camera/frame authority. Metric scale, measurement authority, certified
metrology, mesh generation, Auto selection, hybrid execution, provider
promotion, and permanent global view caps remain unauthorized/false exactly as
before.
