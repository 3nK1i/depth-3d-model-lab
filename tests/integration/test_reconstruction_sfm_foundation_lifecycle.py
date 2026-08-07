from __future__ import annotations

import json
from pathlib import Path

from core.reconstruction import (
    BackendOperationContext,
    DeterministicMockSfmBackend,
    OfflineSfmRequest,
    ReconstructionOperationControl,
    ReconstructionQualityProfile,
    ReconstructionStage,
    materialize_sfm_workspace,
)
from tests.reconstruction_sfm_helpers import SFM_SESSION_ID, make_sfm_ingestion_result


def test_ingestion_workspace_mock_sparse_lifecycle(tmp_path) -> None:
    ingestion = make_sfm_ingestion_result(tmp_path / "sequence", count=8)
    workspace = materialize_sfm_workspace(ingestion, tmp_path / "workspace")
    request = OfflineSfmRequest(
        session_id=SFM_SESSION_ID,
        profile=ReconstructionQualityProfile.EFFICIENT,
        workspace=workspace,
    )
    events = []
    context = BackendOperationContext(
        session_id=SFM_SESSION_ID,
        stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
        profile=ReconstructionQualityProfile.EFFICIENT,
        control=ReconstructionOperationControl(),
        emit_event=events.append,
    )
    result = DeterministicMockSfmBackend().run_sparse_reconstruction(request, context)
    report_path = Path(workspace.workspace_root) / workspace.reports_directory / "mock_sfm_result.json"
    report_path.write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    restored = json.loads(report_path.read_text(encoding="utf-8"))
    assert restored["gauge"]["scale_status"] == "arbitrary"
    assert restored["gauge"]["unit_to_metre"] is None
    assert restored["statistics"]["registered_keyframes"] == len(workspace.images)
    assert events[-1].progress_fraction == 1.0
