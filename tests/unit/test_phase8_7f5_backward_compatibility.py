from pathlib import Path


def test_f5_keeps_prior_f1_f4_public_modules_importable() -> None:
    from core.reconstruction import (
        ReconstructionPipelineOrchestrator,
        PipelinePayloadStore,
        DepthAnything3ProviderExecutor,
        GaussianAppearanceProviderExecutor,
        NerfAppearanceProviderExecutor,
    )
    assert ReconstructionPipelineOrchestrator
    assert PipelinePayloadStore
    assert DepthAnything3ProviderExecutor
    assert GaussianAppearanceProviderExecutor
    assert NerfAppearanceProviderExecutor


def test_f5_does_not_enable_ui_or_add_heavy_production_dependencies() -> None:
    root = Path(__file__).resolve().parents[2]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    scene = (root / "core/reconstruction/persistent_scene.py").read_text(encoding="utf-8")
    assert "reconstruction_ui_enabled" in scene
    assert all(token not in scene for token in ("import torch", "import gsplat", "import nerfstudio", "import tinycudann"))
    assert "depth-anything-3" not in pyproject.lower()
