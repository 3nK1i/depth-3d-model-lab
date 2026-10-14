# Phase 8.9F3 Validation — Real Gaussian Qualification

## Purpose

Phase 8.9F3 proves that the already-qualified external gsplat backend can consume
**real Phase 8.9F2 scene evidence** and produce a real render-capable Gaussian
appearance artifact without changing production reconstruction architecture.

F3 is a release-qualification layer only. It does not add a Gaussian algorithm,
backend, persistence format, UI mode, or production dependency.

## Exact baseline

- branch: `phase8-reconstruction`
- HEAD / origin: `a0366d4e889a227bcdbf19ab60c929814eb96c63`
- Phase 8.9F2 real SfM + DA3 + canonical geometry: PASS

## Truthful F2 → Gaussian bridge

The current production F4 real-backend runner was qualified using a deterministic
F3 fixture. F3 therefore uses a **validation-only bridge** around existing Phase
8 contracts:

1. validate the materialized F2 summary and DA3 evidence;
2. replay the already-qualified DA3 evidence through the existing canonical
   dense/surface executors in memory;
3. require the rebuilt canonical surface counts to match F2 evidence;
4. materialize the existing backend-neutral `NeuralSceneDataset` from the same
   real DA3 RGB frames + DA3 supporting poses;
5. seed Gaussians from the canonical surface using the existing
   `build_gaussian_seed` contract;
6. run the existing D3 held-out/performance benchmark against **gsplat only** in
   the isolated gsplat interpreter.

This does **not** claim that PyCOLMAP poses are fed to DA3 or to the F3 Gaussian
artifact. F2 already established that PyCOLMAP and DA3 are independent evidence
paths and that the F2 canonical surface is derived from DA3 evidence through the
existing canonical contract.

## Authority and scale

Locked for F3:

- canonical geometry remains authoritative geometry;
- Gaussian is `appearance_only`;
- Gaussian never becomes canonical, metric, measurement, inspection/QC, or
  reference-comparison authority;
- arbitrary reconstruction scale remains arbitrary;
- `unit_to_metre=None`;
- `certified_metrology=False`.

## Real campaign

Default F2 input:

`/tmp/phase8_9f2_real_geometry`

Default F3 output:

`/tmp/phase8_9f3_real_gaussian`

The bounded RTX-2060 campaign uses:

- backend: `gsplat_external` only;
- qualified gsplat version: `1.5.3`;
- qualified checkout revision: `937e29912570c372bed6747a5c9bf85fed877bae`;
- 3 real F2/DA3 frames;
- middle frame held out for validation;
- remaining 2 frames used for training;
- 64 optimization steps;
- 3 warm render repetitions;
- CPU LPIPS/quality metrics in the isolated metrics environment;
- measured hardware profile: `rtx_2060_6gb`.

This is bounded release-acceptance evidence, not a production-scale Gaussian
quality ranking.

## Required PASS evidence

The final F3 summary must prove:

- F2 evidence is still PASS and arbitrary-scale;
- the replayed canonical surface matches F2 vertex/face counts;
- real canonical seed count > 0;
- real gsplat held-out benchmark completed;
- real CUDA hardware evidence measured on the physical RTX 2060;
- PSNR / SSIM / LPIPS measured and finite;
- training VRAM / render VRAM / render FPS measured and positive;
- scene artifact exists and is non-empty;
- held-out render manifest and rendered image exist;
- `gaussian_scene_role=appearance_only`;
- `canonical_geometry_authoritative=False` for the Gaussian artifact;
- `certified_metrology=False`;
- production `depth` cannot import the heavy Gaussian runtime;
- no production code is changed by F3.

No minimum PSNR/SSIM/LPIPS threshold is invented in F3. The purpose is to prove
real execution, renderability, provenance, resource measurement, authority and
isolation on the real F2 scene.

## Visual sanity

After PASS inspect:

`/tmp/phase8_9f3_real_gaussian/held_out_comparison.png`

The reference and held-out gsplat render need only be structurally plausible for
this bounded 64-step acceptance workload. F3 does not claim paper-quality 3DGS.

## Delivery scope

Exactly five validation files:

1. `docs/phase_history/PHASE8_9F3_VALIDATION.md`
2. `scripts/run_phase8_9f3_real_gaussian_campaign.py`
3. `scripts/run_phase8_9f3_tests.sh`
4. `tests/integration/test_phase8_9f3_real_gaussian_contract.py`
5. `tests/unit/test_phase8_9f3_real_gaussian_campaign.py`

No `core/`, `app/`, configuration, dependency, or external-checkout file is part
of this delivery.
