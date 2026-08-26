# I3.2 — Second-Scene Production Promotion & Requalification

## Purpose
Promote only the exact I3.1 `d12_sparse` candidate after explicit manual transfer review, regenerate portable canonical Dense RGB geometry, and requalify the second scene without rerunning SfM, DA3, alignment, or fusion.

## Locked safety rules
- I3.1 must be `ready_for_manual_transfer_review` and every transfer safety gate must pass.
- Human transfer review must be explicitly confirmed.
- The saved candidate must be an exact slice of the original second-scene F Extended/Core geometry.
- The scene-relative manifold predicate is rebuilt and must exactly reproduce the I3.1 rejection mask and reference thresholds.
- Extended loss <= 1%, Core loss <= 0.5%, robust diagonal shrink <= 3%, full/robust regression <= 1%.
- Dominant lineage must be 100%, newly detached voxels = 0, removed baseline-dominant voxels = 0.
- Largest-component fraction must not degrade and detached voxel count must not increase.
- Persistent scene must contain 10 verified relative references, be self-contained, and pass relocation restore.

## Authority
I3.2 promotes formal F filtering and canonical G persistence for the second scene only. The SfM gauge remains arbitrary. Metric scale, measurement authority, certified metrology, and surface mesh generation remain disabled.

## Final closure
After automated requalification passes, review promoted fixed views, run Reconstruction UI sanity, then execute the explicit closure command. Successful closure authorizes I4 only; mesh remains blocked until the Offline Dense Quality Campaign reaches its mesh-readiness milestone.
