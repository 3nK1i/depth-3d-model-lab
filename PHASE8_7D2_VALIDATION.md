# Phase 8.7D2 Validation — Real Gaussian Backend Qualification

## Baseline

Apply this delivery only on:

- branch: `phase8-reconstruction`
- HEAD: `c25da4706b2e765ebb501202b818516fb8849504`
- Phase 8.7D1 dual-backend Gaussian foundation already committed and pushed

Phase 8.7D2 qualifies real **external** Gaussian backends. It does not install
Graphdeco, gsplat, CUDA extensions, or their PyTorch stack into the production
`depth` environment.

## Scope

D2 adds:

- pinned isolated environment setup for gsplat and Graphdeco reference 3DGS;
- real CUDA environment/runtime evidence;
- real external gsplat rasterization + backward optimization;
- real bounded Graphdeco `train.py` + `render.py` execution;
- deterministic neural-scene/seed qualification input;
- optimized/render-capable artifact validation;
- exact backend/version/revision, dataset fingerprint, pose, scale, license,
  canonical-geometry, and metrology provenance checks;
- target-hardware qualification reports.

D2 deliberately does **not** claim full scene quality or paper-quality training.
That belongs to 8.7D3 with real datasets, PSNR/SSIM/LPIPS, VRAM, FPS, endurance,
RTX 2060/RTX 5090 profiling, cancellation, and recovery.

## Qualified runtime pins

These are EIA TECH Phase 8.7D2 qualification pins, not a claim that they are the
upstream projects' original reference environments:

- Python `3.10`
- PyTorch `2.7.1`
- torchvision `0.22.1`
- PyTorch CUDA runtime / external CUDA toolkit `12.8`
- minimum CUDA compute capability `7.0`
- gsplat `v1.5.3` / release commit short ID `937e299`
- Graphdeco revision `54c035f7834b564019656c3e3fcc3646292f727d`
- Graphdeco rasterizer compatibility patch `diff-gaussian-rasterization-pr49-cstdint`

Graphdeco's upstream README documents a CUDA-capable GPU with compute capability
7.0+ and a much larger VRAM target for paper-evaluation quality. D2 is a bounded
qualification workload: 512 trusted seed Gaussians, low-resolution frames, short
optimization, and Graphdeco densification disabled. Passing D2 is therefore a
backend/runtime qualification, not a paper-quality benchmark.

The pinned Graphdeco rasterizer revision uses fixed-width integer types without
including `<cstdint>`, which fails on modern GCC toolchains. D2 applies one audited
external-source compatibility patch that adds only `#include <cstdint>` to
`cuda_rasterizer/rasterizer_impl.h`. The setup script verifies the patch SHA-256,
applies it fail-closed after the pinned submodule checkout, and records the patch ID
in runtime qualification provenance. The upstream Graphdeco revision itself remains
unchanged.

## Backend policy

- `GRAPHDECO_REFERENCE_EXTERNAL`: research/evaluation reference backend;
  research-only policy and license provenance from D1 remain enforced.
- `GSPLAT_EXTERNAL`: Apache-2.0 permissive backend path.
- both execute outside `depth`.
- Gaussian artifacts remain `appearance_only`.
- `canonical_geometry_authoritative=False`.
- `certified_metrology=False`.
- Reconstruction UI remains disabled.

## Graphdeco adapter safety

The qualified Graphdeco loader supports undistorted `PINHOLE` input but derives
projection from focal lengths/FOV and does not propagate an off-centre principal
point through its camera contract. The D2 worker therefore rejects off-centre
`cx/cy` instead of silently changing camera geometry. The deterministic D2
qualification input uses a centered principal point.

The worker generates a minimal COLMAP-text scene and an existing
`sparse/0/points3D.ply` from the trusted EIA Gaussian seed. The qualified
Graphdeco reader accepts text camera/extrinsic fallback and consumes an existing
PLY directly. D2 uses `--densify_until_iter 0` so the bounded qualification does
not explode Gaussian count on the RTX 2060.

## gsplat qualification behavior

The real gsplat worker uses `gsplat.rendering.rasterization` with:

- Gaussian quaternions in `wxyz` order;
- `viewmats = inverse(T_reconstruction_from_camera)` (world/reconstruction to camera);
- explicit `K = [fx, fy, cx, cy]` intrinsics;
- trusted seed means/scales held fixed;
- real CUDA forward/backward optimization of color and opacity;
- a real preview render and serialized optimized scene artifact.

This is intentionally a micro-optimization qualification, not the final 3DGS
trainer used for D3 quality benchmarking.

## Delivery files

Exactly 14 project files are part of this overlay:

1. `PHASE8_7D2_VALIDATION.md`
2. `core/reconstruction/__init__.py`
3. `core/reconstruction/gaussian_backends.py`
4. `scripts/setup_gaussian_backends.sh`
5. `scripts/run_gaussian_backend_qualification.py`
6. `scripts/run_phase8_7d2_tests.sh`
7. `scripts/gaussian_workers/common.py`
8. `scripts/gaussian_workers/gsplat_worker.py`
9. `scripts/gaussian_workers/graphdeco_worker.py`
10. `scripts/gaussian_patches/graphdeco_diff_gaussian_rasterization_cstdint.patch`
11. `tests/unit/test_reconstruction_gaussian_backends.py`
12. `tests/unit/test_phase8_7d2_backward_compatibility.py`
13. `tests/integration/test_reconstruction_gaussian_backend_qualification.py`
14. `tests/stress/test_reconstruction_gaussian_backend_qualification_stress.py`

## Dependency-light gates

Run before creating the heavy external environments:

```bash
./scripts/run_phase8_7d2_tests.sh quick
./scripts/run_phase8_7d2_tests.sh integration
./scripts/run_phase8_7d2_tests.sh stress
```

Expected for this delivery:

- quick: `30 passed`
- integration: `1 passed`
- stress: `1 passed`

The qualification runner must also fail closed when the external environments
are absent: it reports `NOT QUALIFIED`, writes a summary, and does not create a
fake optimized artifact.

## External environment setup

From the production repository while `depth` is active:

```bash
./scripts/setup_gaussian_backends.sh --backend both
```

Default external locations:

- conda env `eia-gaussian-gsplat`
- conda env `eia-gaussian-graphdeco`
- checkout root `~/Desktop/eia-gaussian-backends`
- gsplat checkout `~/Desktop/eia-gaussian-backends/gsplat`
- Graphdeco checkout `~/Desktop/eia-gaussian-backends/gaussian-splatting`

Use `--recreate` only for an intentional controlled rebuild. The setup script
refuses to overwrite existing selected environments/checkouts otherwise.

## Real target-hardware qualification

Run on the target NVIDIA machine after setup:

```bash
rm -rf /tmp/phase8_7d2_gaussian_qualification

python scripts/run_gaussian_backend_qualification.py \
  --backend both \
  --gsplat-python /home/eiatech/anaconda3/envs/eia-gaussian-gsplat/bin/python \
  --graphdeco-python /home/eiatech/anaconda3/envs/eia-gaussian-graphdeco/bin/python \
  --graphdeco-root /home/eiatech/Desktop/eia-gaussian-backends/gaussian-splatting \
  --profile efficient \
  --timeout-s 1800
```

Required top-level result:

```text
Phase 8.7D2 real Gaussian backend qualification: PASS
```

Both selected backend reports must have:

- `status=qualified`
- `qualified=true`
- real CUDA execution
- optimization completed
- render written
- optimized + render-capable artifact
- exact runtime/backend pin match
- exact neural-scene dataset fingerprint
- scene artifact and preview present on disk
- `canonical_geometry_authoritative=false`
- `certified_metrology=false`

The final summary is:

`/tmp/phase8_7d2_gaussian_qualification/summary.json`

## Backward/isolation gates

D1 and all established reconstruction quick gates must remain green. Production
`core/reconstruction` must remain free of direct PyCOLMAP, stella_vslam,
Graphdeco, gsplat, Nerfstudio, rasterizer, or simple-knn imports.

No new production dependency is added to `pyproject.toml`.

## Commit gate

Do not commit D2 until all of the following pass on the target machine:

- D2 focused tests;
- real external environment setup;
- real `--backend both` qualification;
- previous quick gates;
- isolation checks;
- full unit/UI/integration/stress regression;
- `python main.py` startup/clean exit;
- `git diff --check` clean;
- exactly the 14 intended D2 files changed (plus any separately preserved user-local unrelated edit);
- branch still `phase8-reconstruction`;
- pre-commit HEAD still `c25da4706b2e765ebb501202b818516fb8849504`.

No Phase 8 release tag is created in D2.
