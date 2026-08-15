# Phase 9.4A — Gaussian Foundation & Runtime Isolation

## Baseline

Apply only on:

- branch: `phase9-digital-twin`
- HEAD/origin: `87930eae31dbfa4d42123d31dfe2b230b04b04f6`
- working tree clean
- Phase 9.3 closed

## Purpose

Phase 9.4A does not reimplement Gaussian Splatting. Phase 8 already established
Gaussian dataset/seed contracts, real isolated gsplat and Graphdeco runtimes,
backend qualification, real RTX-2060 campaigns, process-group supervision, and
static Gaussian preview presentation.

9.4A promotes those proven contracts into one explicit foundation for the future
**interactive** Gaussian path.

## Locked authority boundary

- canonical scene geometry remains geometry authority;
- Gaussian remains `appearance_only`;
- Gaussian is never metric-scale, measurement, inspection/QC, or reference authority;
- `certified_metrology=false`;
- no Gaussian↔NeRF automatic cross-fallback;
- arbitrary canonical scale is never converted to metric by Gaussian;
- gsplat is the permissive Phase 9 interactive candidate;
- Graphdeco remains research/evaluation only under its existing provenance policy.

## Runtime isolation

The default already-qualified external path is:

- interpreter: `~/anaconda3/envs/eia-gaussian-gsplat/bin/python`
- checkout: `~/Desktop/eia-gaussian-backends/gsplat`
- gsplat: `1.5.3`
- qualified checkout revision: `937e29912570c372bed6747a5c9bf85fed877bae`
- Python: `3.10`
- PyTorch: `2.7.1`
- CUDA runtime: `12.8`
- minimum CUDA compute capability: `7.0`

The production `depth` interpreter must not directly expose `gsplat`,
`diff_gaussian_rasterization`, or `simple_knn`.

9.4A capability probing is read-only. It may start short subprocesses in the
isolated interpreter to read versions/CUDA devices, and uses read-only Git
commands against the external checkout. It does **not** train/optimize Gaussians,
render a Gaussian view, create a Gaussian artifact, or start Nerfstudio.

## Process lifecycle policy

Later 9.4 workers must:

- start in a new process session/group;
- never use `shell=True`;
- use a bounded timeout;
- cancel with SIGTERM and escalate to SIGKILL after a grace period;
- leave no orphan external runtime;
- never install the heavy backend into `depth`.

The existing Phase 8 supervised Gaussian lifecycle implementation remains the
reference behavior. 9.4C may wire this policy into the application worker, but
9.4A does not start authoring jobs.

## Validation

Dependency-light:

```bash
./scripts/run_phase9_4a_tests.sh quick
./scripts/run_phase9_4a_tests.sh integration
./scripts/run_phase9_4a_tests.sh backward
python scripts/run_phase9_4a_gaussian_runtime_smoke.py
```

Real read-only runtime probe on the target workstation:

```bash
python scripts/run_phase9_4a_gaussian_runtime_probe.py
```

A qualified machine should end with:

```text
Phase 9.4A isolated gsplat runtime probe: READY
```

The JSON must show:

- `state=ready`;
- `backend_kind=gsplat_external`;
- `interpreter_is_isolated=true`;
- exact qualified checkout revision;
- qualified gsplat/Python/PyTorch/CUDA versions;
- CUDA device evidence from the real machine;
- `production_heavy_modules=[]`;
- `capability_probe_only=true`;
- `training_performed=false`;
- `rendering_performed=false`;
- `canonical_geometry_authoritative=false`;
- `metric_scale_authoritative=false`;
- `measurement_authoritative=false`;
- `qc_authoritative=false`;
- `certified_metrology=false`.

If the external environment is absent, the result must be `UNAVAILABLE`. If the
qualified revision/runtime drifts, the production interpreter is contaminated,
or Graphdeco is proposed as the interactive production backend, the result must
be `REJECTED`. Neither state may trigger fallback or installation.

## Deferred to later Phase 9.4 substeps

- 9.4B: canonical scene → Gaussian authoring-input bridge;
- 9.4C: real isolated Gaussian optimization/backend execution;
- 9.4D: persistent Gaussian representation;
- 9.4E: interactive Gaussian viewer;
- 9.4F: professional controls/scene-layer UX;
- 9.4G: performance/resource lifecycle hardening;
- 9.4H: real-scene qualification and Phase 9.4 closure.

No `v0.9.0` tag or branch merge is created in 9.4A.
