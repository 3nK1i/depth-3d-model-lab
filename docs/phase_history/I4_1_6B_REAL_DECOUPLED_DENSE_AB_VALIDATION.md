# I4.1.6B — Real Decoupled Dense A/B Validation

Purpose: execute the first real DA3 + dense-consistency comparison for the two
I4.1.5/I4.1.6A local-candidate planners without changing the production path.

## Authority rules

- PyCOLMAP global SfM remains the single authoritative geometry frame.
- I4.1.6A PnP-localized frames are observations only.
- Extra frames fit their DA3 scale to the existing arbitrary SfM gauge using
  the exact I4.1.6A PnP-inlier landmarks.
- Existing I1 dense-consistency thresholds and `max_bridge_gap=3` are reused.
- Final fusion remains bounded to at most 40 views; I4.1.6B does not fuse.
- No filtering, persistence, metric scale, measurement authority, or mesh.

## Real A/B plans

1. `uniform_localized_dense_pool`
2. `global_anchors_plus_local_neighbors`

The existing global-only I4 run is reused as the baseline and is not rerun.
Extra DA3 candidates are inferred one frame at a time with one persistent model
load to preserve historical I1/E3 inference semantics while avoiding repeated
model startup.

A production-binding follow-up is authorized only when real evidence gives a
full-span plan with >=80% DA3/SfM alignment acceptance, >=10 connected
pass-qualified views, and a positive gain over the global-only baseline. Dense
threshold relaxation is never authorized by this step.
