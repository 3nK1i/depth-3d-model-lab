from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from core.reconstruction import (
    DenseSceneProfile,
    MetricGeometryStatus,
    NeuralSceneDataset,
    NeuralSceneFrame,
    NerfBackendKind,
    NerfSceneRole,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
    nerf_backend_runtime_pin,
)
from scripts.nerf_workers.common import export_nerfstudio_dataset

ROOT = Path(__file__).resolve().parents[2]


def _dataset(tmp_path: Path) -> NeuralSceneDataset:
    frames = []
    for index in range(3):
        image = (tmp_path / f"f{index}.png").resolve()
        Image.fromarray(np.full((72, 96, 3), 40 + index * 20, dtype=np.uint8), mode="RGB").save(image)
        pose = np.eye(4)
        pose[2, 3] = index * 0.1
        frames.append(
            NeuralSceneFrame(
                frame_id=f"f{index}", image_path=str(image), width=96, height=72,
                fx=80.0, fy=80.0, cx=48.0, cy=36.0,
                T_reconstruction_from_camera=pose, pose_layer="optimized_keyframe",
                camera_profile_id="camera",
            )
        )
    return NeuralSceneDataset(
        session_id="backward-e3", scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        reconstruction_frame_id="reconstruction", camera_profile_id="camera",
        scale_profile_id="arbitrary", metric_geometry_status=MetricGeometryStatus.ARBITRARY,
        scale_status=SfmScaleStatus.ARBITRARY, world_alignment_status=SfmWorldAlignmentStatus.ARBITRARY,
        unit_to_metre=None, relative_scale_uncertainty=None, frames=tuple(frames),
    )


def test_e2_runtime_pin_and_appearance_authority_remain_unchanged() -> None:
    pin = nerf_backend_runtime_pin()
    assert pin.backend_kind is NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL
    assert pin.release_tag == "v1.1.5"
    assert pin.revision_prefix == "6b60855"
    assert NerfSceneRole.APPEARANCE_ONLY.value == "appearance_only"


def test_e2_default_export_split_remains_last_frame_eval(tmp_path: Path) -> None:
    path = export_nerfstudio_dataset(_dataset(tmp_path), tmp_path / "export")
    payload = json.loads(path.read_text())
    assert len(payload["train_filenames"]) == 2
    assert len(payload["val_filenames"]) == 1
    assert payload["val_filenames"] == payload["test_filenames"]
    assert payload["eia_backend_qualification_only"] is True
    assert payload["eia_explicit_heldout_split"] is False


def test_e3_does_not_add_heavy_nerf_runtime_to_production_manifest() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "nerfstudio" not in text
    assert "tiny-cuda-nn" not in text
    assert "tinycudann" not in text
    assert "lpips" not in text


def test_production_reconstruction_core_still_has_no_heavy_nerf_or_metrics_imports() -> None:
    violations = []
    for path in (ROOT / "core/reconstruction").glob("*.py"):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("import nerfstudio", "from nerfstudio", "import tinycudann", "from tinycudann", "import lpips", "from lpips")):
                violations.append(f"{path.name}:{line_no}")
    assert violations == []


def test_e3_reuses_existing_backend_neutral_d3_metrics_worker_without_modifying_it() -> None:
    worker = ROOT / "scripts/gaussian_benchmark_workers/metrics_worker.py"
    text = worker.read_text(encoding="utf-8")
    assert "Backend-neutral PSNR/SSIM/LPIPS evaluator" in text
    assert 'LPIPS_PACKAGE_VERSION = "0.1.4"' in text
    runner = (ROOT / "scripts/run_nerf_real_scene_validation.py").read_text(encoding="utf-8")
    assert '"gaussian_benchmark_workers" / "metrics_worker.py"' in runner
