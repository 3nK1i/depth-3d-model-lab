from __future__ import annotations

import inspect
from pathlib import Path

from core.reconstruction import (
    PersistentReconstructionScene,
    ReconstructionOperationControl,
    ReconstructionPipelineOrchestrator,
    ReconstructionStageExecutionContext,
)


def test_f5_persistent_scene_contract_remains_exported() -> None:
    assert PersistentReconstructionScene.__name__ == "PersistentReconstructionScene"


def test_operation_control_existing_api_remains_present() -> None:
    control = ReconstructionOperationControl()
    assert control.cancellation_requested is False
    assert control.paused is False
    assert control.request_cancel() is True
    assert control.cancellation_requested is True


def test_stage_context_new_control_field_is_optional_for_backward_callers() -> None:
    signature = inspect.signature(ReconstructionStageExecutionContext)
    assert signature.parameters["control"].default is None


def test_g1_core_has_no_heavy_backend_imports() -> None:
    source = (Path(__file__).resolve().parents[2] / "core/reconstruction/engine_hardening.py").read_text()
    forbidden = ("import torch", "import gsplat", "import nerfstudio", "import tinycudann", "import depth_anything_3")
    assert all(value not in source for value in forbidden)


def test_orchestrator_public_run_signature_still_accepts_optional_control() -> None:
    signature = inspect.signature(ReconstructionPipelineOrchestrator.run)
    assert "control" in signature.parameters
    assert signature.parameters["control"].default is None
