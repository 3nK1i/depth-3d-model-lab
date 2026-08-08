from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from core.reconstruction import (
    GAUSSIAN_BACKEND_QUALIFICATION_SCHEMA_VERSION,
    GAUSSIAN_BENCHMARK_SCHEMA_VERSION,
    GAUSSIAN_SPLATTING_SCHEMA_VERSION,
    NEURAL_SCENE_DATASET_SCHEMA_VERSION,
    GaussianSceneArtifact,
    NeuralSceneDataset,
)

ROOT = Path(__file__).resolve().parents[2]


def test_locked_d1_d2_schema_versions_remain_unchanged() -> None:
    assert NEURAL_SCENE_DATASET_SCHEMA_VERSION == 1
    assert GAUSSIAN_SPLATTING_SCHEMA_VERSION == 1
    assert GAUSSIAN_BACKEND_QUALIFICATION_SCHEMA_VERSION == 1
    assert GAUSSIAN_BENCHMARK_SCHEMA_VERSION == 1


def test_d3_does_not_add_benchmark_fields_to_neural_scene_dataset() -> None:
    names = {field.name for field in fields(NeuralSceneDataset)}
    assert "training_frame_ids" not in names
    assert "validation_frame_ids" not in names
    assert "psnr_db" not in names


def test_d3_does_not_add_benchmark_fields_to_gaussian_artifact() -> None:
    names = {field.name for field in fields(GaussianSceneArtifact)}
    assert "psnr_db" not in names
    assert "ssim" not in names
    assert "lpips" not in names
    assert "render_fps" not in names


def test_no_benchmark_dependency_is_added_to_production_environment() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert '"lpips' not in text
    assert '"torchmetrics' not in text
    assert '"scikit-image' not in text
    assert '"gsplat' not in text
    assert '"nerfstudio' not in text


def test_reconstruction_ui_remains_disabled() -> None:
    text = (ROOT / "config/app.yaml").read_text(encoding="utf-8").lower()
    assert "reconstruction_mode_enabled: true" not in text


def test_d2_setup_and_workers_are_not_rewritten_by_step1() -> None:
    assert (ROOT / "scripts/setup_gaussian_backends.sh").is_file()
    assert (ROOT / "scripts/gaussian_workers/gsplat_worker.py").is_file()
    assert (ROOT / "scripts/gaussian_workers/graphdeco_worker.py").is_file()
