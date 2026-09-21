from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.reconstruction_offline_workspace import build_offline_reconstruction_workspace_snapshot
from app.reconstruction_session_model import build_reconstruction_session_model

SESSION = "11111111-1111-4111-8111-111111111111"


def test_i6_1a_contract_composes_existing_g1_session_model_without_replacing_it() -> None:
    source = SimpleNamespace(
        workflow=SimpleNamespace(value="image_sequence"),
        ready=True,
        display_name="Existing reconstruction source",
        detail="12 images",
        path=Path("/tmp/i6_1a/integration-source"),
    )
    session = build_reconstruction_session_model(
        source_selection=source,
        offline_session_id=SESSION,
        offline_state=SimpleNamespace(value="running"),
    )

    snapshot = build_offline_reconstruction_workspace_snapshot(
        session_model=session,
        source_selection=source,
        offline_state=SimpleNamespace(value="running"),
        offline_output_dir="/tmp/i6_1a/integration-output",
        latest_event={
            "protocol": "eia.reconstruction.offline.v1",
            "event": "progress",
            "stage": "source_ingestion",
            "message": "Offline ingestion started.",
            "progress_fraction": 0.0,
        },
    )

    assert snapshot.session_model is session
    assert snapshot.session_id == SESSION
    assert snapshot.lifecycle == "running"
    assert snapshot.current_stage == "source_ingestion"
    assert snapshot.source_ready is True
    assert snapshot.production_auto_authorized is False


def test_i6_1a_contract_remains_passive_and_does_not_import_pyqt_or_auto_engine() -> None:
    module_text = Path("app/reconstruction_offline_workspace.py").read_text(encoding="utf-8")
    assert "PyQt" not in module_text
    assert "i5_auto" not in module_text
    assert "AutoPolicy" not in module_text
    assert "dense_provider_implementation_contracts" not in module_text
    assert "OfflineReconstructionController" not in module_text
    assert "ReconstructionController" not in module_text
