from __future__ import annotations

from pathlib import Path

from core.reconstruction import (
    ENGINE_HARDENING_PHASE,
    PersistentReconstructionScene,
    ReconstructionEngineProcessSpec,
    ReconstructionPipelineOrchestrator,
)


def test_g1_hardening_contract_remains_exported():
    assert ENGINE_HARDENING_PHASE == "8.7G1"
    assert ReconstructionEngineProcessSpec.__name__ == "ReconstructionEngineProcessSpec"


def test_f5_persistent_scene_contract_remains_exported():
    assert PersistentReconstructionScene.__name__ == "PersistentReconstructionScene"


def test_orchestrator_still_exists_under_same_public_name():
    assert ReconstructionPipelineOrchestrator.__name__ == "ReconstructionPipelineOrchestrator"


def test_g2_core_has_no_heavy_backend_imports():
    source = (Path(__file__).resolve().parents[2] / "core/reconstruction/engine_checkpoint.py").read_text()
    forbidden = ("import torch", "import gsplat", "import nerfstudio", "import tinycudann", "import depth_anything_3")
    assert all(value not in source for value in forbidden)
