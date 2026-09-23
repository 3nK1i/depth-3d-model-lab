# I6.8.5B — Reconstruction method and eligibility contract

Date: 2026-09-22
Source basis: user-supplied committed-source `depth_lab_i6_8_5a_source_bb1d132.zip` (external archive SHA256 `34047dcc53398a6ee5ed359eabd59a80cc323ac461ec32fd26ec965cdbca9463`; the ZIP contains no `.git`).
Delivery: contracts, focused tests, runner, and this document only. No production launch, UI or provider changes.

## Scope and non-negotiable product requirements

I6.8.5A determined that there is no unified *executable* reconstruction-method selector. The I6.4 dense-provider catalog expresses provider intent/qualification, not execution; this new dependency-light method contract describes distinct user workflows, their *actual* existing launch point, required evidence and output type. It does not override or duplicate the I6.4 catalog. It has no stateful selection, runtime probing, importing of backends, worker launching, GPU heuristics or UI controls. I6.8.5C/D must bind to existing controllers and reuse I6.4 evidence where appropriate; a method is never mapped to a different provider without explicit user action.

True depth prediction + **TSDF** fusion is a mandatory I7 implementation and v1.0 acceptance item, **not** an optional candidate or a synonym for the existing voxel-averaged point accumulator. Do not enable a TSDF action before signed-distance-field integration, bounded volume storage, surface extraction, canonical pose/depth/scale provenance, persistence where applicable, and geometric/operational qualification are implemented. The I6.8.5 UI must report this requirement honestly until completion. Do not silently remove TSDF from the catalog or future I13 matrices.

## Audited production references / currently supported scopes

| Catalog method | Existing reference / status | Inputs and output | Preconditions, limitations |
| --- | --- | --- | --- |
| `colmap_mvs` | `app.reconstruction_dense_job_controller.ExplicitDenseJobController.start` is currently wired to **Build dense (COLMAP MVS)** | Image sequence or recorded-video-origin **admitted sparse** session → independently verified dense points | Requires successful, admitted, arbitrary-scale sparse result and qualified isolated PyCOLMAP runtime. `start` explicitly refuses any provider ID other than COLMAP MVS. |
| `da3_quick` | `app.da3_canonical_integration_controller.DA3CanonicalIntegrationController.start`, existing **Integrate DA3** | Image directory or recorded video → bounded supporting depth/relative-pose geometry | `DA3CanonicalIntegrationConfiguration` allows 2–8 frames per job, default **3**. Not full sequence coverage. Qualified external DA3 interpreter/repository must be available. |
| `da3_full_scene` | `app.reconstruction_da3_full_scene.build_full_scene_da3_job`, existing **Build Full Scene** | Recorded video only → full-scene DA3 geometry evidence | Image-directory coverage is **not** implemented in this current entry point and must not be enabled by a renamed button. Uses bounded overlapping 3-view windows; external DA3 runtime required. |
| `voxel_point_fusion` | `core.reconstruction.dense_fusion`, `core.reconstruction.da3_dense_provider` | Aligned depth evidence → voxel-fused **points** | Existing internal component; not a standalone Reconstruction-page method launch. Not TSDF, no extracted signed-distance isosurface. |
| `gaussian_training` | `core.reconstruction.gaussian_optimization_execution`, existing persisted appearance viewer in `app/widgets/reconstruction_page.py` | Qualified canonical scene + surface seed → Gaussian **appearance**, not a metric mesh | Training is not wired as a Reconstruction-page launch, despite existing backend and visualization. Remains integration-pending; cannot run by selecting a representation. |
| `tsdf` | Mandatory I7 implementation; no TSDF integrator found in inspected `core/`, `app/` or `scripts/` | Aligned depth + camera poses → bounded signed-distance volume + surface/mesh | Explicit implementation-and-qualification gate. Existing voxel fusion cannot satisfy it. |

## Contract behavior

`reconstruction_method_catalog()` returns immutable, unambiguous method specifications. `assess_reconstruction_method(method, context)` returns a *fail-closed advisory* assessment with explicit reason codes. An existing entry point is executable **only when** the caller has independently verified the source, required sparse/scene evidence and exact method runtime, and no conflicting job is active. Method IDs must be explicit; `Auto` and silent fallback are prohibited at this boundary. Unwired/internal/future methods cannot become executable by supplying an optimistic context.

The assessment is not an execution authorization or provider evidence promotion. I6.8.5C/D must revalidate against current state at launch and display clear unavailable reasons rather than silently start COLMAP. Selection does not certify camera poses, metric scale, dense admission, geometry, measurement, QC, or metrology. Current PyCOLMAP canonical-camera policy is preserved; a future policy change requires a separate, evidence-backed decision. Cancellation is documented only for existing explicit actions. Pause and resume are **not** advertised; I6.9 must qualify those operations per backend/stage.

The `runtime_verified_methods` input is **caller-supplied verified evidence**, not a check for GPU name, total VRAM, Python executable mere existence, or a performance guarantee. I6.11 will own qualified adaptive hardware settings. I6.8.5B neither promises 57-image DA3 feasibility on an RTX 2060 nor runs model inference.

## Acceptance / next steps

- B: Pure contract, strict output/provider distinction, TSDF mandatory entry, explicit source/prerequisite/runtimes, no execution side effects. Run `bash scripts/run_i6_8_5b_tests.sh` plus the small backward-compatible I6.4A contract regression, compile, shell syntax and staged diff check.
- C: Reuse existing controller launches; investigate verified full-sequence image-directory DA3 execution, and Gaussian's current canonical-surface prerequisites. Avoid routing all choices to MVS; unavailable paths stay blocked with reasons.
- D: Integrate an ergonomic method selector and explicit result-type/status labels in the existing Reconstruction page without cluttering Runtime Status. Existing pending/inspection selectors must retain their own semantics.
- E: End-to-end qualification per *enabled* method and representative bounded data, including session isolation, correctness/quality and provenance. TSDF remains an explicit mandatory I7 and I13 acceptance item.

No existing running reconstruction is interrupted by B; apply the overlay and run validation only after the user's current job reaches a terminal state or is otherwise safely idle.
