# I2.5.2 — Extremum Local-Geometry / Manifold Diagnosis

## Purpose

I2.5.1 isolated only 657 full-scene extremum drivers and 184 stubborn survivors.  Most stubborn survivors are far-peripheral and many are already Core, while only a minority are low-diversity.  I2.5.2 therefore diagnoses the missing **local 3D geometry / manifold attachment** evidence rather than trying another view-removal or diversity-only filter.

This milestone is diagnostic only.  It does not change source geometry, F policy, G persistence, SfM, DA3, metric/measurement authority, or mesh authority.

## Evidence

For every retained Extended voxel I2.5.2 measures:

- KNN spacing in voxel units;
- covariance-eigenvalue linearity / planarity / scattering;
- nearest distance to a strong Core multi-view manifold anchor;
- number of strong anchors within a local voxel-radius;
- alignment between the local principal axis and the mean supporting camera ray.

Strong manifold anchors are defined only from existing reconstruction evidence: Core, support >= 3, at least two nearby high-support anchors, and not an I2.5.1 extremum driver.  I2 peripheral/far labels are not used to create the manifold.

A central Core non-driver population supplies diagnostic quantile references.  A combined local-geometry signature requires manifold separation plus either sparse local geometry or strongly ray-aligned linear structure.  This is an analysis label, not a production filter.

## Required validation

```bash
bash scripts/run_i2_5_2_tests.sh focused
bash scripts/run_i2_5_2_tests.sh quick
bash scripts/run_i2_5_2_tests.sh affected
```

Real diagnosis:

```bash
python scripts/run_i2_5_2_extremum_local_geometry_manifold_diagnosis.py \
  --i1-dir /tmp/eia_i1_offline_dense_campaign_recovered \
  --i2-dir /tmp/eia_i2_full_video_quality \
  --i2-5-1-dir /tmp/eia_i2_5_1_spatial_extremum_survivor_diagnosis \
  --output-dir /tmp/eia_i2_5_2_extremum_local_geometry_manifold_diagnosis
```

The exact retained-view observation accounting must remain exact.  No production filtering or I2/mesh authorization is permitted by this step alone.
