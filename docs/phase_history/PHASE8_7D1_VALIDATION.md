# Phase 8.7D1 — Gaussian Splatting Foundation, Dual Backend Policy & Isolation

Required baseline:

- Branch: `phase8-reconstruction`
- HEAD: `094dcfeafe5169494da46da29589998cb295de84`
- Working tree clean and synchronized with `origin/phase8-reconstruction`

## Why 8.7D is split into bounded substeps

Gaussian Splatting introduces CUDA-heavy training/runtime dependencies and must not destabilize the production `depth` environment. Phase 8.7D is therefore split into:

1. **8.7D1 — Foundation, dual-backend provenance & isolation** (this package): backend-neutral neural-scene dataset, quality-aware Gaussian seed, external-worker contract, immutable backend/license provenance, environment probes, deterministic external mock.
2. **8.7D2 — Real isolated backend qualification**: qualify separate CUDA/PyTorch environments for both the Graphdeco reference implementation and `gsplat`, then integrate real optimization/render outputs through the same worker contract.
3. **8.7D3 — Real-scene validation & hardening**: train/render bounded real captures, compare backends, verify recovery/cancellation/resource cleanup, and close 8.7D.

No Phase 8.7D substep may install Gaussian dependencies into the production `depth` environment.

## Dual backend policy

Phase 8.7D supports two real external backends without mixing their licensing or intended-use semantics:

### Graphdeco reference backend

- Backend kind: `graphdeco_reference_external`
- Upstream: `https://github.com/graphdeco-inria/gaussian-splatting`
- Upstream license: `https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md`
- Purpose in EIA TECH: scientific/reference implementation for industrial research, evaluation, reproducibility, benchmarking and publication.
- Contract usage scope: `research_evaluation_only`
- `commercial_use_permitted_by_backend_license=False`
- `explicit_commercial_consent_required=True`
- Redistribution must retain the upstream license/attribution requirements.

The upstream license expressly includes academic and industrial research users, but restricts the software to non-commercial research/evaluation unless the licensors provide prior explicit consent. Publishing the EIA TECH project source does not remove or replace those upstream restrictions.

### gsplat backend

- Backend kind: `gsplat_external`
- Upstream: `https://github.com/nerfstudio-project/gsplat`
- License: Apache-2.0 (`https://github.com/nerfstudio-project/gsplat/blob/main/LICENSE`)
- Purpose in EIA TECH: permissive real backend and future production-compatible path, subject to normal Apache-2.0 compliance.
- Contract usage scope: `permissive_general_use`
- `commercial_use_permitted_by_backend_license=True`
- `explicit_commercial_consent_required=False`

The backend/license mapping is canonical in code. A Graphdeco result cannot be relabeled as permissive through worker JSON.

## Architectural invariants

- Canonical QC geometry remains the 8.6C/8.6D/8.7A/8.7B/8.7C chain.
- Gaussian scenes are **appearance-only** and must never become authoritative inspection/metrology geometry.
- `certified_metrology=False` remains mandatory.
- Pose convention remains `T_reconstruction_from_camera`.
- Arbitrary/provisional metric scale semantics are preserved but do not make the Gaussian representation metrology-authoritative.
- Reconstruction UI remains disabled until Phase 8.8.
- PyCOLMAP and stella isolation remain unchanged.
- Graphdeco, `gsplat`, Nerfstudio, Graphdeco CUDA extensions and related backends are not direct production dependencies and are not imported by `core/reconstruction`.
- Real Gaussian workers must use a separate interpreter by default.
- Every worker request and artifact carries backend/license/usage provenance.

## Focused tests

```bash
./scripts/run_phase8_7d1_tests.sh quick
./scripts/run_phase8_7d1_tests.sh integration
./scripts/run_phase8_7d1_tests.sh stress
python scripts/run_gaussian_splatting_foundation_smoke.py
```

Expected focused counts for this corrected package are `46 passed`, `1 passed`, and `1 passed`.

The smoke must finish with:

```text
Phase 8.7D1 Gaussian splatting foundation smoke: PASS
```

Critical semantics:

- Gaussian seed positions are exact selected canonical mesh vertices.
- Source mesh positions remain unchanged.
- External development worker executes out-of-process.
- Development result is `seed_only`, not falsely `optimized`.
- `render_capable=False` for the development seed artifact.
- `canonical_geometry_authoritative=False`.
- `scene_role=appearance_only`.
- `certified_metrology=False`.
- Graphdeco provenance is fixed to `research_evaluation_only`.
- Graphdeco commercial use is not represented as permitted without explicit additional licensor consent.
- `gsplat` provenance remains the separate Apache-2.0 permissive path.
- Reconstruction UI remains disabled.

## Environment probes for 8.7D2

Do **not** install either real backend into `depth`.

Probe a separate `gsplat` interpreter with:

```bash
python scripts/check_gaussian_splatting_environment.py \
  --backend gsplat \
  --python /absolute/path/to/isolated/gsplat/python
```

Probe a separate Graphdeco interpreter and checkout with:

```bash
python scripts/check_gaussian_splatting_environment.py \
  --backend graphdeco-reference \
  --python /absolute/path/to/isolated/graphdeco/python \
  --graphdeco-root /absolute/path/to/graphdeco/gaussian-splatting
```

The Graphdeco probe requires the checkout to retain `LICENSE.md` and the expected reference files. It also probes the external CUDA rasterizer/KNN extensions without importing them in production Python.

If you intentionally probe the current production `depth` interpreter, `NOT READY` is the desired result in 8.7D1. Real backends will be installed/qualified only in 8.7D2 external environments.

## Full target regression

```bash
./scripts/run_phase8_7d1_tests.sh full
python main.py
```

Only the same established optional perception integration skips are acceptable. Reconstruction must remain disabled.

Do not commit until focused validation, backward gates, full regression, isolation checks, and final Git hygiene checks pass.
