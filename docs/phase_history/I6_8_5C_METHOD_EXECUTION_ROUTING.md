# I6.8.5C — Existing reconstruction-method execution routing

Status: **implementation candidate; NOT workstation-validated, NOT committed.**
Source baseline: `ad4d2abfe76a8e54c64a4d80bbce1f8315901b73` (`hotfix/v0.9.0-video-calibration-context`), captured in the externally SHA256-verified 2026-10-09 authoritative handoff.

## Change boundary

- Add `app/reconstruction_method_execution.py`, a Qt-free, fail-closed dispatcher consuming I6.8.5B's `assess_reconstruction_method()`. It delegates **only** to the exact named existing method callback. The method's eligibility and runtime proof are checked before any callback. A backend error propagates; no retry with another provider, Auto, or fallback.
- Add `ReconstructionPage.start_selected_reconstruction_method()` as an *integration API*, **not** as a new visible UI control. It composes current (not cached) source readiness, active reconstruction/Digital Twin job ownership, sparse admission, current sparse source/session identity, and explicit runtime verification evidence into `MethodEligibilityContext`.
- For COLMAP MVS, reuse the existing `ReconstructionController.start_explicit_post_sparse_dense()` and established resource release; the controller/isolated dense worker remain the authority for worker launch, dense verification, session match, and final admission. A new source selection must not reuse old sparse cameras.
- For DA3 Quick, reuse `ReconstructionController.start_selected_da3_canonical()`; its configuration/isolated worker/source validation and existing runtime probe remain authoritative. Quick continues to be bounded 2–8 frames, not a full-scene method.
- For DA3 Full Scene, reuse the existing page-owned `build_full_scene_da3_job()` and `start_digital_twin_processing_job()`; Digital Twin resource coordination, heavy-worker control, admission and cancellation are not forked or replaced. Recorded video only.
- Voxel point fusion remains internal-only; Gaussian method integration remains pending; true signed-distance TSDF remains a mandatory I7 implementation. These methods always fail closed regardless of any optimistic runtime claims.

## Runtime-evidence boundary

`runtime_verified_methods` passed to the new page seam must be independently and method-specifically verified by the caller. It **defaults empty** and cannot be inferred from GPU name, the existence of a Python executable, a path string or an unrelated DA3 method. The existing DA3 workers independently re-probe/validate their external runtime at execution; that later worker check is **not** a substitute for supplying valid prelaunch evidence. This candidate does **not** invent a new GUI-thread, blocking, external GPU runtime probe. I6.8.5D must consume this seam only after a concrete qualified evidence-acquisition flow is designed and tested. The existing explicit buttons are intentionally unmodified in this C patch; their backward-compatible start semantics are preserved.

This patch does not alter UI layout, default action mappings, controller worker lifecycles, persistent scene/camera ownership, output scale, TSDF/Gaussian implementations, or I6.4 provider intent.

## Acceptance gates

1. Verify expected branch/HEAD and clean working tree before applying patch.
2. Apply only the patch's six intended files. Never stage automatically.
3. Run `bash scripts/run_i6_8_5c_tests.sh` in conda `depth` after normal app jobs are idle. Qt-backed UI tests require PyQt6/pytest-qt on the actual workstation. Do not count skipped Qt tests as UI acceptance.
4. Verify the exact file scope, `git diff --check`, source compile, method-routing regressions and no unintended worktree files.
5. Review workstation output before any staging, commit or push. Actual GPU/DA3/PyCOLMAP execution and E2E admission have **not** been performed in this source-only candidate; they require an explicit later acceptance gate.

## Local sandbox qualification

The first source-only candidate passed targeted tests in the extracted archive against existing I6.8.5B and 9.3H contracts. The sandbox lacks PyQt6; Qt integration tests were intentionally added for workstation validation, **not** represented as locally passing. This is not a completed or closed I6.8.5C checkpoint.
