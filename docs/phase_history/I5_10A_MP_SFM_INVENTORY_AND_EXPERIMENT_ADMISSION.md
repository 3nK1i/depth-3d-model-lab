# I5.10A MP-SfM Inventory and Experiment Admission

## Scope

I5.10A is an execution-free inventory and scientific-admission step. Repository phase history explicitly couples MP-SfM to **matching-strategy evaluation**. The current production-native sparse-SfM implementation remains the existing external PyCOLMAP backend; I5.10A does not introduce a new SfM backend or a new matcher.

No SfM, bundle adjustment, MVS, dense-provider inference, Auto, Hybrid, or recovery experiment is executed in I5.10A. No threshold, production default, camera model, feature strategy, or global-view policy is modified.

## Repository inspection

The inspected sparse-SfM contract exposes exactly two matching strategies:

- `sequential`;
- `exhaustive`.

The existing worker uses COLMAP SIFT, `CameraMode.SINGLE`, `SIMPLE_RADIAL`, the pinned PyCOLMAP 4.1.1 environment, and the established sparse-SfM quality policy. Sequential matching uses configurable overlap, quadratic overlap, and loop detection disabled. Exhaustive matching uses the existing PyCOLMAP exhaustive matcher. Vocabulary-tree, spatial, transitive, or newly invented matching modes are outside I5.10.

## Why the experiment is narrow

The strongest repository-supported matching-policy question is the frozen I5.4 KITTI-style forward-motion case:

- I5.4A admitted the first 60 native-order source images as one immutable contiguous window;
- profile was `efficient`;
- image-sequence FPS was 10.0;
- I5.4B used sequential matching with overlap 10;
- feature extraction and matching completed;
- incremental mapping was attempted;
- PyCOLMAP produced no sparse model;
- therefore zero cameras registered and forward-axis geometry could not be authoritatively revalidated;
- I5.4C preserved the result as `REJECTED`, outside the current production sparse-SfM envelope for that admitted case.

By contrast, the difficult-handheld Horse replay already completed global sparse SfM under exhaustive matching and degraded later at local dense-consistency connectivity. Horse therefore does not establish matching policy as the primary sparse bottleneck and is retained as historical context rather than rerun in I5.10C.

Current Castle successes under exhaustive matching are also context only. Cross-scenario success cannot establish matching-policy causality.

## Admitted counterfactual

The only real candidate admitted for I5.10C is:

- scenario: the exact frozen I5.4 60-image KITTI-style window;
- historical baseline: frozen I5.4B sequential matching, overlap 10;
- baseline rerun: **not authorized**;
- counterfactual candidate: existing exhaustive matching;
- attempts: exactly one immutable candidate attempt;
- same source bytes/window: required;
- same `efficient` profile: required;
- same 60-keyframe maximum: required;
- same 10.0 image-sequence FPS interpretation: required;
- same COLMAP SIFT feature strategy: required;
- same camera mode/model and sparse quality policy: required;
- no retry-until-pass, threshold relaxation, result-driven retuning, or default promotion.

I5.10B must bind the exact I5.4A/I5.4B/I5.4C evidence chain, revalidate the complete source identity and selected 60-image window, probe PyCOLMAP 4.1.1 plus both matching APIs, and freeze the exhaustive candidate command. I5.10B remains execution-free.

## Comparison semantics

Allowed evidence comparison is limited to quantities that remain meaningful across independent arbitrary-gauge SfM reconstructions:

- whether a sparse model exists;
- registered camera count and registration ratio;
- landmarks, observations, mean track length, and mean reprojection error;
- sparse quality status and reasons;
- forward-axis dominance only if the candidate produces a valid camera solution;
- established scale-invariant parallax / baseline-to-range diagnostics.

Absolute poses, absolute translation magnitudes, or absolute sparse geometry coordinates may not be compared across the independent SfM gauges. No metric-scale or metrology claim is created. Runtime observations may be recorded but are not a ranking axis in I5.10.

## Outcome semantics

If the exhaustive candidate produces an accepted sparse model, I5.10 may conclude only that matching-policy sensitivity was observed on this admitted KITTI-style case. It does not prove exhaustive matching is universally better or justify an automatic production-default change.

If the candidate is degraded, the partial sensitivity and remaining quality limits are preserved. If the candidate is rejected/no sparse model, the evidence says no exhaustive rescue was observed on the admitted case. Runtime/integrity failure makes the experiment incomplete and is not converted into a scientific quality rejection.

## Authority

PyCOLMAP remains the camera/scene-frame authority for every evaluated sparse reconstruction. Each run remains arbitrary-gauge monocular SfM. No candidate reconstruction automatically replaces the production default or another authoritative run. No provider ranking, universal winner, physical metric scale, measurement authority, certified metrology, or mesh authority is created.

## Roadmap transition

A passing I5.10A artifact authorizes only **I5.10B — MP-SfM runtime/evidence preflight**. I5.10C is not authorized until B is frozen. I5.10D remains the evidence-only qualification/closure step. I5.11 OOM/disk/crash/recovery, I5.12 Hybrid if justified, I5.13 first evidence-driven Auto policy, and I5.14 final I5 closure remain deferred.
