from __future__ import annotations

from pathlib import Path
import sys

from core.reconstruction import (
    GaussianBackendKind,
    GaussianOptimizationProfile,
    GaussianWorkerRequest,
    current_process_gaussian_heavy_modules,
    materialize_gaussian_input_from_canonical_scene,
    materialize_gaussian_input_from_da3_persistent_scene,
    read_neural_scene_dataset_manifest,
)
from tests.phase9_4b_helpers import make_phase9_4b_da3_scene, make_phase9_4b_explicit_scene


def test_4b_canonical_bridge_feeds_existing_phase8_gaussian_worker_contract(tmp_path: Path) -> None:
    scene, frames, _poses = make_phase9_4b_explicit_scene(tmp_path / "fixture")
    result = materialize_gaussian_input_from_canonical_scene(
        scene, frames, output_dir=tmp_path / "bridge"
    )
    request = GaussianWorkerRequest.from_dataset(
        request_id="phase9-4b-worker-contract",
        backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL,
        profile=GaussianOptimizationProfile.EFFICIENT,
        dataset=result.dataset,
        dataset_manifest_path=result.manifest_path,
        seed_path=(tmp_path / "future-seed.npz").resolve(),
        output_dir=(tmp_path / "future-gaussian-output").resolve(),
        maximum_input_frames=result.frame_count,
    )
    assert request.dataset_fingerprint_sha256 == result.dataset_fingerprint_sha256
    assert request.dataset_manifest_path == result.manifest_path
    assert request.scene_profile is result.dataset.scene_profile
    assert current_process_gaussian_heavy_modules() == ()


def test_4b_da3_persistent_scene_materializes_self_contained_gaussian_dataset(tmp_path: Path) -> None:
    before = set(sys.modules)
    scene, _paths, _poses, _intrinsics = make_phase9_4b_da3_scene(tmp_path / "fixture")
    result = materialize_gaussian_input_from_da3_persistent_scene(
        scene, output_dir=tmp_path / "bridge"
    )
    loaded = read_neural_scene_dataset_manifest(result.manifest_path)
    assert loaded.frame_count == 3
    assert all(Path(frame.image_path).parent.name == "images" for frame in loaded.frames)
    assert all(Path(frame.image_path).is_file() for frame in loaded.frames)
    assert result.dataset_fingerprint_sha256
    assert result.training_started is False
    assert result.rendering_started is False
    assert current_process_gaussian_heavy_modules() == ()
    assert not ({"gsplat", "diff_gaussian_rasterization", "simple_knn"} & (set(sys.modules) - before))
