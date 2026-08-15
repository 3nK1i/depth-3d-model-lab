from __future__ import annotations

from pathlib import Path

from core.reconstruction import (
    GaussianBackendKind,
    GaussianOptimizationProfile,
    GaussianWorkerRequest,
    build_gaussian_seed_from_canonical_arrays,
    materialize_gaussian_input_from_da3_persistent_scene,
    write_gaussian_seed_npz,
)
from tests.phase9_4b_helpers import make_phase9_4b_da3_scene
from tests.phase9_4c_helpers import ready_gaussian_runtime, tetrahedral_canonical_arrays


def test_4c_4b_dataset_is_directly_compatible_with_existing_gsplat_worker_request(tmp_path: Path) -> None:
    scene, _paths, _poses, _intrinsics = make_phase9_4b_da3_scene(tmp_path / "scene")
    bridge = materialize_gaussian_input_from_da3_persistent_scene(
        scene,
        output_dir=(tmp_path / "dataset").resolve(),
    )
    vertices, faces, colors = tetrahedral_canonical_arrays()
    seed = build_gaussian_seed_from_canonical_arrays(vertices, faces, colors)
    seed_path = write_gaussian_seed_npz(seed, (tmp_path / "seed.npz").resolve())
    request = GaussianWorkerRequest.from_dataset(
        request_id="phase9-4c-integration",
        backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL,
        profile=GaussianOptimizationProfile.EFFICIENT,
        dataset=bridge.dataset,
        dataset_manifest_path=bridge.manifest_path,
        seed_path=seed_path,
        output_dir=(tmp_path / "worker-output").resolve(),
        maximum_input_frames=8,
    )
    assert request.dataset_fingerprint_sha256 == bridge.dataset_fingerprint_sha256
    assert request.backend_kind is GaussianBackendKind.GSPLAT_EXTERNAL
    assert request.maximum_input_frames == 8
    assert bridge.dataset.metric_geometry_status.value == "arbitrary"
    assert bridge.dataset.unit_to_metre is None


def test_4c_runtime_and_input_bridge_do_not_import_or_start_heavy_backend(tmp_path: Path) -> None:
    runtime = ready_gaussian_runtime(tmp_path / "runtime")
    scene, _paths, _poses, _intrinsics = make_phase9_4b_da3_scene(tmp_path / "scene")
    bridge = materialize_gaussian_input_from_da3_persistent_scene(
        scene,
        output_dir=(tmp_path / "dataset").resolve(),
    )
    assert runtime.ready is True
    assert runtime.training_performed is False
    assert runtime.rendering_performed is False
    assert bridge.training_started is False
    assert bridge.rendering_started is False
    assert bridge.canonical_geometry_authoritative is False
    assert bridge.metric_scale_authoritative is False
