from __future__ import annotations

from pathlib import Path

from core.reconstruction import (
    NerfBackendKind,
    NerfOptimizationProfile,
    NerfWorkerRequest,
    materialize_nerf_input_from_da3_persistent_scene,
)
from tests.phase9_4b_helpers import make_phase9_4b_da3_scene


def test_5c_5b_dataset_is_directly_compatible_with_existing_nerfstudio_worker_request(tmp_path: Path) -> None:
    scene, _paths, _poses, _intrinsics = make_phase9_4b_da3_scene(tmp_path / "scene")
    bridge = materialize_nerf_input_from_da3_persistent_scene(scene, output_dir=tmp_path / "dataset")
    request = NerfWorkerRequest.from_dataset(
        request_id="phase9-5c-integration",
        backend_kind=NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL,
        profile=NerfOptimizationProfile.EFFICIENT,
        dataset=bridge.dataset,
        dataset_manifest_path=bridge.manifest_path,
        output_dir=(tmp_path / "worker-output").resolve(),
        maximum_input_frames=8,
        qualification_training_steps=64,
    )
    assert request.dataset_fingerprint_sha256 == bridge.dataset_fingerprint_sha256
    assert request.maximum_input_frames == 8
    assert request.qualification_training_steps == 64
    assert bridge.dataset.metric_geometry_status.value == "arbitrary"
    assert bridge.dataset.unit_to_metre is None


def test_5c_reuses_existing_phase8_worker_without_modifying_it() -> None:
    root = Path(__file__).resolve().parents[2]
    worker = root / "scripts/nerf_workers/nerfstudio_worker.py"
    text = worker.read_text(encoding="utf-8")
    assert "Nerfstudio/Nerfacto worker for Phase 8.7E2" in text
    assert "torch.cuda.reset_peak_memory_stats()" in text
    assert "training_completed_without_oom" in text
    assert "preview.png" in text
