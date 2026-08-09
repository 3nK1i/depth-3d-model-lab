"""Deterministic development-only orchestration providers for Phase 8.7F1 tests/smokes."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.reconstruction.orchestration import (
    ReconstructionPipelineArtifact,
    ReconstructionStageExecutionContext,
)


@dataclass(slots=True)
class DeterministicMockStageExecutor:
    provider_id: str
    fail_count: int = 0
    payload_prefix: str = "mock://phase8-7f1"
    calls: list[str] = field(default_factory=list)

    def execute(self, context: ReconstructionStageExecutionContext) -> ReconstructionPipelineArtifact:
        if context.provider.provider_id != self.provider_id:
            raise RuntimeError("Mock executor/provider identity mismatch")
        self.calls.append(context.planned_stage.stage.value)
        if self.fail_count > 0:
            self.fail_count -= 1
            raise RuntimeError("deterministic_mock_failure")
        contract = context.planned_stage.contract
        return ReconstructionPipelineArtifact(
            artifact_id=f"{context.request.request_id}:{contract.stage.value}:{self.provider_id}",
            session_id=context.request.session_id,
            stage=contract.stage,
            kind=contract.output_kind,
            authority=contract.output_authority,
            provider_id=self.provider_id,
            payload_ref=f"{self.payload_prefix}/{context.request.request_id}/{contract.stage.value}",
            metadata={
                "development_mock": True,
                "required_input_count": len(context.required_inputs),
                "optional_input_count": len(context.optional_inputs),
            },
        )
