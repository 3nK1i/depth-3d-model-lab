# I5.10B — MP-SfM Runtime + Evidence Preflight

I5.10B is an execution-free preflight for the single matching-strategy counterfactual admitted by I5.10A. It does **not** run PyCOLMAP sparse reconstruction, bundle adjustment, MVS, a dense provider, Auto, or Hybrid logic.

## Frozen scientific question

The historical I5.4 case used the exact first 60 images of the operator-confirmed KITTI-style ordered source with `sequential` matching, overlap 10, the `efficient` profile, maximum 60 keyframes, and nominal sequence FPS 10.0. Feature extraction and matching were reached, but incremental mapping produced no sparse model. I5.10A admits one counterfactual only: existing `exhaustive` matching on the exact same 60 source bytes.

I5.10B revalidates the I5.4A/B/C chain, rehashes the full historical source and every selected source image, probes the isolated PyCOLMAP environment, verifies PyCOLMAP 4.1.1 plus `match_sequential` and `match_exhaustive`, verifies current production worker/source anchors, and freezes the I5.10C command template. No source image is transformed or materialized by B.

## No-Git/Gitea source baseline

The project now continues without Git/Gitea until release or an explicit later GitHub migration decision. I5.10B therefore binds the immutable post-I5.10A source checkpoint by SHA256 as a first-class predecessor. Current repository implementation anchors are also SHA-verified before formal preflight is emitted.

## Candidate freeze

The I5.10C candidate remains:

- scenario: `kitti_forward_motion`
- input: exact frozen 60-image I5.4 window
- matching: `exhaustive`
- feature strategy: COLMAP SIFT
- profile: `efficient`
- max keyframes: 60
- nominal image sequence FPS: 10.0
- attempt count: exactly 1
- baseline rerun: false
- retry-until-pass: false
- threshold relaxation: false
- production-default modification: false
- output directory: new and immutable

For 60 images, exhaustive pairing contains 1,770 unordered image pairs. This number is experiment scope, not a performance score.

If CUDA is available in the frozen PyCOLMAP environment, the candidate uses it. If no CUDA device is visible, B records an explicit CPU fallback and freezes `--cpu` in the command; no silent device substitution is permitted.

## Authority and comparison limits

PyCOLMAP remains camera/global-frame authority. Independent monocular SfM runs retain independent arbitrary gauges. Allowed comparison is registration coverage, sparse health, reprojection/track evidence, and scale-invariant geometry diagnostics. Absolute cross-run pose/translation/geometry and metric-scale comparisons remain prohibited.

A successful exhaustive candidate can establish only matching-policy sensitivity on this one admitted case. It cannot establish universal exhaustive superiority and cannot change the production default in I5.10C.

I5.10B also freezes the I5.10C evidence contract before execution: a new immutable attempt directory, the exact candidate command, start/finish timestamps, return code, console-log SHA256, post-run source/worker integrity, and preservation of either a real SfM report or the recognized no-sparse-model quality rejection. Runtime/integrity failure remains distinct from a scientific quality rejection and cannot be overwritten by retrying.

I5.10B authorizes I5.10C only after all bindings, source identities, runtime APIs, repository anchors, receipt checks, and the immutable source checkpoint are verified. OOM/disk/crash failure injection is explicitly deferred to I5.11; I5.10B does not adaptively reduce the 60-image experiment for convenience. I5.10D and I5.11–I5.14 remain deferred.
