from __future__ import annotations

import json
from pathlib import Path
import sys

from core.reconstruction import (
    NerfBackendKind,
    NerfOptimizationProfile,
    NerfWorkerRequest,
    current_process_nerf_heavy_modules,
    materialize_nerf_input_from_da3_persistent_scene,
    read_neural_scene_dataset_manifest,
)
from scripts.nerf_workers.common import export_nerfstudio_dataset
from tests.phase9_4b_helpers import make_phase9_4b_da3_scene


def test_5b_canonical_bridge_feeds_existing_nerf_worker_contract(tmp_path: Path) -> None:
    before = set(sys.modules)
    scene, _paths, _poses, _intrinsics = make_phase9_4b_da3_scene(tmp_path / "fixture")
    result = materialize_nerf_input_from_da3_persistent_scene(scene, output_dir=tmp_path / "bridge")
    loaded = read_neural_scene_dataset_manifest(result.manifest_path)
    transforms = export_nerfstudio_dataset(
        loaded,
        tmp_path / "nerfstudio",
        training_frame_ids=tuple(frame.frame_id for frame in loaded.frames[:-1]),
        validation_frame_ids=(loaded.frames[-1].frame_id,),
    )
    request = NerfWorkerRequest(
        request_id="phase9-5b-worker-contract",
        backend_kind=NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL,
        profile=NerfOptimizationProfile.EFFICIENT,
        dataset_manifest_path=result.manifest_path,
        dataset_fingerprint_sha256=result.dataset_fingerprint_sha256,
        output_dir=str((tmp_path / "future-nerf-output").resolve()),
        scene_profile=result.dataset.scene_profile,
        maximum_input_frames=result.frame_count,
    )
    payload = json.loads(transforms.read_text(encoding="utf-8"))
    assert request.dataset_fingerprint_sha256 == result.dataset_fingerprint_sha256
    assert len(payload["frames"]) == 3
    assert current_process_nerf_heavy_modules() == ()
    assert not ({"nerfstudio", "tinycudann", "torch"} & (set(sys.modules) - before))
