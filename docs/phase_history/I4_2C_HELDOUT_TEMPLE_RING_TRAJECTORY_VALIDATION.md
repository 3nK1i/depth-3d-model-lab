# I4.2C — Held-Out TempleRing Trajectory Validation

## Scope

I4.2C validates the independently estimated I4.2B PyCOLMAP camera trajectory
against the Middlebury TempleRing per-view calibration that I4.2A quarantined.
It is validation-only and does not rerun or modify SfM/BA.

## Evidence separation

- I4.2B must already be `PASS`, with 47/47 views and fixed shared intrinsics.
- The authoritative `cameras.bin`, `images.bin`, and `points3D.bin` hashes must
  still match I4.2B exactly.
- A read-only PyCOLMAP worker opens only the authoritative sparse model and has
  no access to Middlebury R/t.
- The parent process then opens the checksum-frozen held-out validation
  manifest and recomputes every ground-truth camera centre as `C = -R^T t`.
- `templeR_ang.txt` is not consumed.

## Alignment and metrics

One proper, reflection-free Sim(3) is fitted from all 47 SfM camera centres to
all 47 held-out camera centres using Umeyama least squares:

`x_gt = s * R_gt_from_sfm * x_sfm + t_gt`

Only camera centres participate in the fit. Held-out camera rotations are not
used to estimate the similarity. After the centre-only Sim(3) is frozen, SfM
camera rotations are expressed in the held-out world axes and independently
compared against the per-view Middlebury rotations.

Reported metrics include:

- camera-centre RMSE / median / p95 / max in benchmark coordinate units;
- the same centre errors normalized by the held-out median ring radius;
- camera rotation median / mean / p95 / max in degrees.

The default I4.2 engineering qualification gates are declared before the real
held-out result is inspected:

- normalized centre RMSE <= 0.02;
- normalized centre p95 <= 0.03;
- normalized centre max <= 0.06;
- rotation median <= 1 degree;
- rotation p95 <= 2 degrees;
- rotation max <= 5 degrees.

These are internal I4.2 qualification gates, not official Middlebury scoring,
not certified calibration limits, and not metrology thresholds.

## Authority locks

I4.2C never:

- writes the Sim(3) back into the authoritative sparse reconstruction;
- rewrites or refines I4.2B cameras;
- reruns SfM or bundle adjustment;
- starts MVS or DA3;
- changes production filtering/manifold policy;
- claims metric scale or authoritative measurement;
- generates a mesh;
- authorizes the I4.3 provider architecture.

A passing I4.2C authorizes only I4.2D: TempleRing MVS dense reconstruction from
the unchanged I4.2B authoritative sparse model.
