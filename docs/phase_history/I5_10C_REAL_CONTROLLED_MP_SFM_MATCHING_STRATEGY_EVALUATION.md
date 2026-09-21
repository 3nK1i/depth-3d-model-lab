# I5.10C — Real Controlled MP-SfM Matching-Strategy Evaluation

I5.10C executes the single counterfactual admitted by I5.10A and frozen by I5.10B. The historical I5.4 baseline is reused as immutable evidence: the exact 60-image KITTI-style window with sequential matching (overlap 10) reached incremental mapping but produced no sparse model and was rejected. I5.10C changes only the matching policy to the existing exhaustive matcher.

## Frozen execution

- exact same 60 source image bytes and order
- PyCOLMAP 4.1.1 production worker
- `CameraMode.SINGLE`, `SIMPLE_RADIAL`, COLMAP SIFT
- efficient profile, max keyframes 60, image-sequence FPS 10.0
- exhaustive matching, 1,770 possible image pairs
- one real attempt only
- baseline is not rerun
- no threshold relaxation, adaptive image reduction, retry-until-pass, or production-default modification

## Terminal semantics

`pass`, `degraded`, and `rejected` are completed scientific outcomes. The known `PyCOLMAP incremental mapping produced no sparse model` path is preserved as a legitimate rejection without a real-SfM report. Unexpected process failures, report-contract failures, source drift, staged-byte drift, or worker/repository-anchor drift are `runtime_integrity_failure` and are not interpreted as reconstruction quality.

## Authority

PyCOLMAP remains camera/global-frame authority for the candidate reconstruction. Its monocular SfM gauge is arbitrary and independent of the historical baseline. No metric scale, metrology, absolute cross-run translation/pose comparison, universal matching-strategy winner, production promotion, Auto policy, or Hybrid policy is established here.

I5.10D is authorized only after one scientifically terminal I5.10C attempt with intact source/staged/worker evidence.
