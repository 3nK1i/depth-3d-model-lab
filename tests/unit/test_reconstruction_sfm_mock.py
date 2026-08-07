from __future__ import annotations

from core.reconstruction import (
    BackendOperationContext,
    DeterministicMockSfmBackend,
    OfflineSfmRequest,
    ReconstructionOperationControl,
    ReconstructionQualityProfile,
    ReconstructionStage,
    SfmScaleStatus,
    materialize_sfm_workspace,
)
from tests.reconstruction_sfm_helpers import SFM_SESSION_ID, make_sfm_ingestion_result


def _run_mock(tmp_path):
    ingestion = make_sfm_ingestion_result(tmp_path / "sequence", count=5)
    workspace = materialize_sfm_workspace(ingestion, tmp_path / "workspace")
    request = OfflineSfmRequest(
        session_id=SFM_SESSION_ID,
        profile=ReconstructionQualityProfile.BALANCED,
        workspace=workspace,
    )
    captured = []
    context = BackendOperationContext(
        session_id=SFM_SESSION_ID,
        stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
        profile=ReconstructionQualityProfile.BALANCED,
        control=ReconstructionOperationControl(),
        emit_event=captured.append,
    )
    backend = DeterministicMockSfmBackend()
    result = backend.run_sparse_reconstruction(request, context)
    backend.shutdown()
    return result, captured


def test_mock_backend_is_deterministic_and_arbitrary_scale(tmp_path) -> None:
    first, events = _run_mock(tmp_path / "a")
    second, _ = _run_mock(tmp_path / "b")
    assert first.gauge.scale_status is SfmScaleStatus.ARBITRARY
    assert first.gauge.unit_to_metre is None
    assert [item.as_dict() for item in first.cameras] == [item.as_dict() for item in second.cameras]
    assert [item.as_dict() for item in first.landmarks] == [item.as_dict() for item in second.landmarks]
    assert events[-1].progress_fraction == 1.0
    assert first.statistics.registered_keyframes == 5


def test_mock_backend_capabilities_do_not_claim_metric_scale() -> None:
    backend = DeterministicMockSfmBackend()
    assert not backend.capabilities.supports_metric_scale
    assert backend.capabilities.supports_cpu
    assert not backend.capabilities.requires_cuda
