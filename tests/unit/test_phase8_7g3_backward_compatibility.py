from __future__ import annotations

import inspect

from core.reconstruction import (
    ReconstructionEngineProcessSpec,
    ReconstructionPipelineOrchestrator,
    ReconstructionEnduranceCampaignReport,
)


def test_g3_does_not_change_orchestrator_public_defaults():
    sig = inspect.signature(ReconstructionPipelineOrchestrator.run)
    assert sig.parameters["control"].default is None
    assert tuple(sig.parameters["resume_stage_results"].default) == ()
    assert sig.parameters["stage_commit_hook"].default is None


def test_g1_supervisor_contract_remains_available():
    sig = inspect.signature(ReconstructionEngineProcessSpec)
    assert "cancel_after_s" in sig.parameters
    assert "terminate_grace_s" in sig.parameters


def test_g3_report_keeps_g4_and_ui_boundaries_explicit():
    fields = ReconstructionEnduranceCampaignReport.__dataclass_fields__
    assert fields["full_regression_deferred_to"].default == "8.7G4"
    assert fields["reconstruction_ui_enabled"].default is False
