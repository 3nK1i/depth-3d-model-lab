# I4.3E1 — Metric3D v2 production provider adapter

## Scope

I4.3E1 binds the existing Metric3D v2 model integration to the canonical I4.3A
`single_image_depth` provider family. It is an additive offline reconstruction
adapter only; it does not redesign or replace the existing interactive model
adapter.

This substep intentionally covers **Metric3D v2 only**. UniDepthV2, MoGe-2,
Depth Pro, Depth Anything V2, and Marigold remain separate I4.3E substeps because
their native depth, intrinsics, confidence/uncertainty, and geometry semantics
differ.

## Locked authority

- PyCOLMAP SfM remains the sole authoritative global/camera frame.
- Every registered view is available by default; no permanent global view cap is introduced.
- Adaptive view selection remains deferred to I4.3F.
- Metric3D native metric depth is model evidence only and is **not** promoted to physical canonical metric scale.
- Metric3D estimated/virtual-FOV intrinsics are model evidence only and are **not** used for canonical projection.
- Canonical projection uses bundle-adjusted PyCOLMAP poses and the authoritative SfM camera model.
- Per-view robust scale-only sparse/depth alignment maps Metric3D depth into the arbitrary SfM gauge.
- The provider remains `candidate`; no provider selection, promotion, Auto policy, hybrid authorization, measurement authority, certified metrology, or mesh authority is created.

## Execution

A lazy external worker transport was added at
`scripts/reconstruction_single_image_depth_provider_worker.py`. In I4.3E1 its
allow-list contains only `metric3d-v2`. Heavy imports/model loading happen only
inside worker execution, not when the worker module is imported.

The Metric3D provider:

1. validates the explicit I4.3A provider selection and authoritative SfM frame;
2. verifies registered staged-image hashes;
3. sends every registered view to the external Metric3D worker;
4. validates native Metric3D semantics (`metric`, estimated intrinsics, confidence);
5. aligns each depth map to the corresponding SfM keyframe using sparse correspondences;
6. fuses only accepted aligned views with PyCOLMAP poses/camera geometry;
7. persists a canonical candidate point cloud plus execution/provenance summary;
8. verifies source-image and SfM authority digests are unchanged.

The current dense fusion contract, like I4.3C/D1, expects one authoritative SfM
camera model per provider run. Multiple camera-model groups fail closed rather
than being silently reinterpreted.

## Added files

- `core/reconstruction/metric3d_v2_dense_provider.py`
- `scripts/reconstruction_single_image_depth_provider_worker.py`
- `scripts/run_i4_3e1_metric3d_v2_binding_smoke.py`
- `scripts/run_i4_3e1_tests.sh`
- `tests/i4_3e1_helpers.py`
- `tests/unit/test_i4_3e1_metric3d_v2_dense_provider.py`
- `tests/integration/test_i4_3e1_metric3d_v2_provider_contract.py`
- `tests/stress/test_i4_3e1_metric3d_v2_provider_stress.py`

## Validation in the packaged inspection copy

- I4.3D3 backward matrix: `54 passed`.
- Existing Metric3D/model-contract backward tests + I4.3E1 focused/stress tests: `69 passed`.
- Single-image worker lazy-import probe: PASS.
- Python compilation and shell syntax: PASS.

These results validate the packaged implementation structure only. The user's
RTX workstation remains authoritative for the real installed Metric3D runtime
binding gate.

## Authoritative workstation closure gate

Before I4.3E1 is declared closed, run:

1. `bash scripts/run_i4_3e1_tests.sh all`
2. execution-free `run_i4_3e1_metric3d_v2_binding_smoke.py` against the real passing I4.3A and I4.3D3 snapshots;
3. Python compilation, shell syntax, conflict-marker/trailing-whitespace/final-newline checks;
4. read-only source-control status while Gitea is unavailable.

No Git mutation is authorized while Gitea remains unavailable.

## Next

If the authoritative workstation gate passes, I4.3E1 closes as a candidate
provider binding and I4.3E2 may begin with UniDepthV2. Cross-provider quality
comparison and any provider promotion remain deferred to I4.3G/I5.
