from __future__ import annotations

from core.reconstruction.dense_provider_architecture import (
    DenseProviderDescriptor,
    DenseProviderImplementation,
    DenseProviderQualificationState,
)
from core.reconstruction.orchestration import ReconstructionProviderExecution


def available_descriptor(implementation: DenseProviderImplementation, provider_id: str = "f2-test-provider") -> DenseProviderDescriptor:
    return DenseProviderDescriptor(
        provider_id=provider_id,
        implementation=implementation,
        execution=ReconstructionProviderExecution.EXTERNAL_PROCESS,
        backend_version="test-backend",
        adapter_version="i4.3f2-test",
        qualification_state=DenseProviderQualificationState.CANDIDATE,
        available=True,
        unavailable_reason="",
    )
