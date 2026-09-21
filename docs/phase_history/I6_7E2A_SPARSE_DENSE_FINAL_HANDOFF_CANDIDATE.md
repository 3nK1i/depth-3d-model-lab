# I6.7E2A — Sparse → Dense final-artifact handoff candidate (initial I6.7E2 increment)

## Frozen baseline

Source: immutable I6.7D archive `depth_3d_model_lab_post_i6_7d_source_20260919_121022_265721.tar.gz`, SHA256 `d3cf16e1a3a96371b2f42d976b957dc0e93d99d2f71f638a25c5f8d4cdf3da6d`. Frozen I6.7A/B/C/D source and the completed I6.7E1 read-only artifact scanner remain unchanged. I6.7E1's workstation source probe verified 1,686,160 real RGB points in the frozen external `fused.ply` artifact; that *separate* I6.7B evidence does not, by itself, constitute a product-admitted offline session.

## Why E2 is divided

The actual One Start job currently stops at a sparse SfM result. It exposes an I6.2C admitted sparse-result handoff but does **not** execute a dense provider in the normal offline product workflow. An I6.7B standalone provider smoke generated a full artifact and a bounded progressive stream without accessing that production handoff. Connecting these immediately to persistence or the GPU viewer would bypass missing run/session/frame and file-integrity admission checks.

## E2A implementation and bounds

`prepare_dense_final_handoff` accepts only a previously admitted I6.2C partial sparse handoff, corresponding persisted `offline_result.json` and `sfm_result.json`, one typed arbitrary-scale PyCOLMAP `OfflineSfmResult`, an explicitly selected concrete COLMAP-MVS request and decision (no Auto, fallback or override), and a completed/non-degraded I6.7B provider execution envelope. It verifies the exact sparse session, persisted SfM digest, selected source-view population, reconstruction-frame binding, run UUID, request fingerprint, selected provider/result identity and prohibition on any progressive/final/metric promotion. A per-run, non-symlinked provider output must be placed at:

`<already-admitted-offline-output>/dense_runs/<dense-run-uuid>/provider_colmap_mvs/dense/fused.ply`

The caller supplies a completed-provider artifact SHA256 from the provider's completed output evidence, **not** from the provisional snapshot. The E1 bounded full-vertex preflight scans/rehashes the exact completed PLY and validates finite geometry and RGB, then produces an immutable `QualifiedDenseFinalHandoffCandidate` containing the complete geometry reference, digest, vertex count and arbitrary SfM frame binding. Neither successful status snapshots nor the 4,096-point preview can substitute for these checks. A failed GUI snapshot sink does not, by itself, invalidate an otherwise valid completed artifact. This candidate is not a persistent scene or a committed final-result admission; it must be revalidated at subsequent consumption to reject time-of-check/time-of-use drift.

**No One Start execution wiring, worker/status protocol, controller, UI, viewport, persistent scene, canonical camera pose, metric scale, measurement, mesh, scientific provider parameters or Auto policy is modified in E2A.** The 4,096-point provisional display bound and the PyCOLMAP camera/global-frame authority are unchanged. This pure/CPU full-PLY validation must run off the Qt UI thread when integrated. Full-density GPU rendering is a distinct later gate, not a side effect of artifact eligibility.

## Validation

New unit tests include passing completion and rejection of foreign/stale session, malformed handoff, missing camera authority, fallback/Auto, mismatched frame/provider/view population/fingerprint/source SHA, claimed provisional promotion, corrupt artifact SHA, tampered persisted SfM/offline result, escaped/symlinked geometry, missing RGB. An adapter integration test runs the existing MVS wrapper with a **fake external worker** and verifies that an actual completed adapter result yields a candidate independently of the provisional sink. No real PatchMatch/fusion is performed for these tests. Run `bash scripts/run_i6_7e2a_tests.sh focused` then `closure` on the workstation; `closure` additionally enforces the unchanged frozen-source hashes and storage floor and runs the full unit suite. Preserve any failing evidence; do not retry-until-pass.

## Remaining I6.7E work — not completed by this package

E2B must wire an explicitly selected dense provider into the real One Start worker/controller lifecycle, transport bounded provisional updates, preserve cancellation and fail-soft preview semantics, collect completed-provider output SHA independently, and commit final-result admission/persistence transactionally using this candidate and the existing production scene contracts. The frozen I6.7B evidence cannot be presented as a live production result without these bindings. Complete dense-geometry GPU rendering and performance/UX qualification follow the explicit final-result handoff, with I6.7F the phase's professional dense UX/performance/closure gate. Do not authorize I6.7F or GitHub migration from passing this E2A candidate alone.

## I6.7E2A architecture correction (full-unit regression)

The first focused gate passed, but the full unit suite correctly rejected the candidate module under `core/reconstruction/` because it imports the I6.2C application-layer sparse handoff. The same, byte-identical candidate implementation is now located at `app/reconstruction_dense_final_handoff.py`; its typed application boundary remains explicit and the reconstruction core stays independent of `app` and `PyQt6`. Only test import paths and the test-runner compile/architecture gate were changed. The original failing test and its evidence must remain intact. No production controller/worker/UI/viewport or frozen I6.7A–D source was modified; no final admission/persistence or full-density GPU visualization was enabled.
