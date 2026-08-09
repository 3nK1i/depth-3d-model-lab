from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from core.reconstruction import (
    DenseSceneProfile,
    MetricGeometryStatus,
    NERFSTUDIO_E2_CUDA,
    NERFSTUDIO_E2_PYTHON,
    NERFSTUDIO_E2_RTX2060_COMPUTE_CAPABILITY,
    NERFSTUDIO_E2_TORCH,
    NERFSTUDIO_E2_TORCHVISION,
    TINY_CUDA_NN_E2_REVISION,
    NerfBackendKind,
    NerfEnvironmentProbe,
    NerfOptimizationProfile,
    NerfWorkerRequest,
    NeuralSceneDataset,
    NeuralSceneFrame,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
    nerfstudio_camera_to_world_from_eia_pose,
)
from scripts.nerf_workers.common import export_nerfstudio_dataset, profile_settings
import scripts.run_nerf_backend_qualification as e2_qualification


def _dataset(tmp_path: Path, count: int = 3) -> NeuralSceneDataset:
    frames = []
    for index in range(count):
        image = (tmp_path / f"frame-{index}.ppm").resolve()
        image.write_bytes(b"P6\n4 3\n255\n" + bytes([80 + index, 100, 120]) * 12)
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = index * 0.1
        frames.append(
            NeuralSceneFrame(
                frame_id=f"frame-{index}",
                image_path=str(image),
                width=4,
                height=3,
                fx=4.0,
                fy=4.0,
                cx=2.0,
                cy=1.5,
                T_reconstruction_from_camera=pose,
                pose_layer="optimized_keyframe",
                camera_profile_id="camera",
            )
        )
    return NeuralSceneDataset(
        session_id="e2-unit",
        scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        reconstruction_frame_id="reconstruction",
        camera_profile_id="camera",
        scale_profile_id="arbitrary",
        metric_geometry_status=MetricGeometryStatus.ARBITRARY,
        scale_status=SfmScaleStatus.ARBITRARY,
        world_alignment_status=SfmWorldAlignmentStatus.ARBITRARY,
        unit_to_metre=None,
        relative_scale_uncertainty=None,
        frames=tuple(frames),
    )


def test_e2_runtime_recipe_is_explicit_and_rtx2060_targeted() -> None:
    assert NERFSTUDIO_E2_PYTHON == "3.10"
    assert NERFSTUDIO_E2_TORCH == "2.1.2"
    assert NERFSTUDIO_E2_TORCHVISION == "0.16.2"
    assert NERFSTUDIO_E2_CUDA == "11.8"
    assert NERFSTUDIO_E2_RTX2060_COMPUTE_CAPABILITY == (7, 5)
    assert TINY_CUDA_NN_E2_REVISION == "b3473c81396fe927293bdfd5a6be32df8769927c"


def test_eia_to_nerfstudio_pose_flips_camera_yz_only() -> None:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    converted = np.asarray(nerfstudio_camera_to_world_from_eia_pose(pose))
    assert np.allclose(converted[:3, 0], [1.0, 0.0, 0.0])
    assert np.allclose(converted[:3, 1], [0.0, -1.0, 0.0])
    assert np.allclose(converted[:3, 2], [0.0, 0.0, -1.0])
    assert np.allclose(converted[:3, 3], [1.0, 2.0, 3.0])
    assert np.allclose(converted[3], [0.0, 0.0, 0.0, 1.0])


def test_eia_to_nerfstudio_pose_rejects_invalid_shape() -> None:
    with pytest.raises(Exception):
        nerfstudio_camera_to_world_from_eia_pose(np.eye(3))


def test_exporter_writes_explicit_train_and_test_split(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=3)
    transforms = export_nerfstudio_dataset(dataset, tmp_path / "ns")
    payload = json.loads(transforms.read_text())
    assert payload["camera_model"] == "OPENCV"
    assert payload["orientation_override"] == "none"
    assert len(payload["frames"]) == 3
    assert len(payload["train_filenames"]) == 2
    assert len(payload["val_filenames"]) == 1
    assert payload["test_filenames"] == payload["val_filenames"]
    assert all((transforms.parent / frame["file_path"]).is_file() for frame in payload["frames"])
    converted = np.asarray(payload["frames"][0]["transform_matrix"])
    assert converted[1, 1] == -1.0
    assert converted[2, 2] == -1.0


def test_exporter_single_frame_keeps_valid_train_and_test_split(tmp_path: Path) -> None:
    payload = json.loads(export_nerfstudio_dataset(_dataset(tmp_path, 1), tmp_path / "single").read_text())
    assert payload["train_filenames"] == payload["test_filenames"]
    assert len(payload["train_filenames"]) == 1


def test_e2_profile_settings_are_bounded() -> None:
    efficient = profile_settings(NerfOptimizationProfile.EFFICIENT)
    high = profile_settings(NerfOptimizationProfile.HIGH_QUALITY)
    assert efficient["train_rays_per_batch"] == 512
    assert high["train_rays_per_batch"] == 2048
    assert efficient["eval_num_rays_per_chunk"] < high["eval_num_rays_per_chunk"]


def test_request_can_carry_bounded_e2_training_steps(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    request = NerfWorkerRequest.from_dataset(
        request_id="e2",
        backend_kind=NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL,
        profile=NerfOptimizationProfile.EFFICIENT,
        dataset=dataset,
        dataset_manifest_path=(tmp_path / "dataset.json").resolve(),
        output_dir=(tmp_path / "out").resolve(),
        maximum_input_frames=16,
        qualification_training_steps=16,
    )
    assert request.qualification_training_steps == 16
    assert request.as_dict()["qualification_training_steps"] == 16


def test_request_rejects_unbounded_e2_training_steps(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    with pytest.raises(Exception):
        NerfWorkerRequest.from_dataset(
            request_id="e2",
            backend_kind=NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL,
            profile=NerfOptimizationProfile.EFFICIENT,
            dataset=dataset,
            dataset_manifest_path=(tmp_path / "dataset.json").resolve(),
            output_dir=(tmp_path / "out").resolve(),
            maximum_input_frames=16,
            qualification_training_steps=257,
        )


def test_environment_probe_requires_ns_render_for_e2_readiness() -> None:
    kwargs = dict(
        python_executable=str(Path(sys.executable).resolve()),
        available=True,
        nerfstudio_version="1.1.5",
        torch_version="2.1.2+cu118",
        tinycudann_available=True,
        ns_train_available=True,
        cuda_available=True,
        cuda_device_count=1,
        cuda_device_names=("NVIDIA GeForce RTX 2060",),
        cuda_compute_capabilities=((7, 5),),
        cuda_total_vram_mib=(6144.0,),
        torch_cuda_version="11.8",
    )
    assert NerfEnvironmentProbe(**kwargs, ns_render_available=True).backend_ready is True
    assert NerfEnvironmentProbe(**kwargs, ns_render_available=False).backend_ready is False


def test_environment_probe_fails_closed_on_incomplete_cuda_evidence() -> None:
    with pytest.raises(Exception):
        NerfEnvironmentProbe(
            python_executable=str(Path(sys.executable).resolve()),
            available=True,
            nerfstudio_version="1.1.5",
            torch_version="2.1.2",
            tinycudann_available=True,
            ns_train_available=True,
            ns_render_available=True,
            cuda_available=True,
            cuda_device_count=1,
            cuda_device_names=("GPU",),
        )


def test_setup_resume_mode_preserves_partial_isolated_environment() -> None:
    text = (Path(__file__).resolve().parents[2] / "scripts/setup_nerfstudio_backend.sh").read_text(encoding="utf-8")
    assert "--resume" in text
    assert "--recreate and --resume are mutually exclusive" in text
    assert "resume_runtime_prerequisites=PASS" in text
    assert "--resume requires existing conda env" in text


def test_setup_resolves_nonempty_environment_prefix_and_cuda_driver_stub() -> None:
    text = (Path(__file__).resolve().parents[2] / "scripts/setup_nerfstudio_backend.sh").read_text(encoding="utf-8")
    assert "import sys; print(sys.prefix)" in text
    assert "sed '/^[[:space:]]*$/d'" in text
    assert '[[ -n "$ENV_PREFIX" && -d "$ENV_PREFIX" ]]' in text
    assert "targets/x86_64-linux/lib/stubs" in text
    assert 'LIBRARY_PATH="$CUDA_STUB_DIR"' in text
    assert "Never add stubs to" in text
    assert "LD_LIBRARY_PATH" in text


def test_setup_resume_skips_rebuilding_importable_tinycudann() -> None:
    text = (Path(__file__).resolve().parents[2] / "scripts/setup_nerfstudio_backend.sh").read_text(encoding="utf-8")
    assert "python -c 'import tinycudann'" in text
    assert "tiny_cuda_nn_runtime=already_installed" in text
    assert 'pip install --no-build-isolation "$TCNN_ROOT/bindings/torch"' in text


def test_nerfstudio_install_keeps_pep517_build_isolation_enabled() -> None:
    text = (Path(__file__).resolve().parents[2] / "scripts/setup_nerfstudio_backend.sh").read_text(encoding="utf-8")
    marker = 'echo "=== Install pinned Nerfstudio ==="'
    nerfstudio_block = text.split(marker, 1)[1].split('conda run -n "$ENV_NAME" python -m pip check', 1)[0]
    assert 'python -m pip install "$NERFSTUDIO_ROOT" "numpy==1.26.4"' in nerfstudio_block
    assert "--no-build-isolation" not in nerfstudio_block
    assert "transitive source" in text
    assert "fpsample" in text
    assert "scikit-build-core + pybind11" in text


def _qualified_rtx2060_probe(total_vram_mib: float) -> NerfEnvironmentProbe:
    return NerfEnvironmentProbe(
        python_executable="/tmp/eia-nerf-nerfstudio/bin/python",
        available=True,
        nerfstudio_version="1.1.5",
        torch_version="2.1.2+cu118",
        tinycudann_available=True,
        ns_train_available=True,
        ns_render_available=True,
        cuda_available=True,
        cuda_device_count=1,
        cuda_device_names=("NVIDIA GeForce RTX 2060",),
        cuda_compute_capabilities=((7, 5),),
        cuda_total_vram_mib=(total_vram_mib,),
        torch_cuda_version="11.8",
    )


def test_rtx2060_preflight_accepts_real_cuda_visible_memory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ns_root = tmp_path / "nerfstudio"
    tcnn_root = tmp_path / "tiny-cuda-nn"
    ns_root.mkdir()
    tcnn_root.mkdir()

    def fake_revision(root: Path) -> str:
        if root == ns_root:
            return "6b60855003011b2ca23c2fe3f8e2ca6314c69924"
        if root == tcnn_root:
            return TINY_CUDA_NN_E2_REVISION
        return ""

    monkeypatch.setattr(e2_qualification, "_git_revision", fake_revision)
    reasons = e2_qualification._preflight_reasons(
        probe=_qualified_rtx2060_probe(5744.5625),
        nerfstudio_root=ns_root,
        tcnn_root=tcnn_root,
        hardware_profile="rtx_2060_6gb",
    )
    assert reasons == []


def test_rtx2060_preflight_still_rejects_materially_smaller_cuda_memory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ns_root = tmp_path / "nerfstudio"
    tcnn_root = tmp_path / "tiny-cuda-nn"
    ns_root.mkdir()
    tcnn_root.mkdir()

    def fake_revision(root: Path) -> str:
        if root == ns_root:
            return "6b60855003011b2ca23c2fe3f8e2ca6314c69924"
        if root == tcnn_root:
            return TINY_CUDA_NN_E2_REVISION
        return ""

    monkeypatch.setattr(e2_qualification, "_git_revision", fake_revision)
    reasons = e2_qualification._preflight_reasons(
        probe=_qualified_rtx2060_probe(5400.0),
        nerfstudio_root=ns_root,
        tcnn_root=tcnn_root,
        hardware_profile="rtx_2060_6gb",
    )
    assert reasons == ["rtx2060_cuda_visible_vram_below_profile"]


def test_nerfstudio_single_test_render_accepts_flat_rgb_png(tmp_path: Path) -> None:
    from scripts.nerf_workers.nerfstudio_worker import _find_dataset_render_rgb_png

    render_root = tmp_path / "rendered"
    flat = render_root / "test" / "rgb.png"
    flat.parent.mkdir(parents=True)
    flat.write_bytes(b"\x89PNG\r\n\x1a\nEIA")
    assert _find_dataset_render_rgb_png(render_root) == flat


def test_nerfstudio_dataset_render_accepts_nested_rgb_png(tmp_path: Path) -> None:
    from scripts.nerf_workers.nerfstudio_worker import _find_dataset_render_rgb_png

    render_root = tmp_path / "rendered"
    nested = render_root / "test" / "rgb" / "00004_frame.png"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"\x89PNG\r\n\x1a\nEIA")
    assert _find_dataset_render_rgb_png(render_root) == nested
