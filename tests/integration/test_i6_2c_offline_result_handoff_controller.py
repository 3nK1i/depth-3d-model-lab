from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.application_controller import ApplicationController
from app.reconstruction_controller import ReconstructionController
from app.reconstruction_offline_result_handoff import OfflineResultHandoffAvailability
from app.widgets.original_view import OriginalSourceView
from app.widgets.source_bar import SourceBar


@pytest.fixture
def controller(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    source_bar = SourceBar()
    original_view = OriginalSourceView()
    qtbot.addWidget(source_bar)
    qtbot.addWidget(original_view)
    application = ApplicationController(source_bar, original_view)
    value = ReconstructionController(application, parent=application)
    yield value
    value.shutdown()
    application.shutdown()


def _payload(root: Path, sid: str) -> dict[str, object]:
    visual = root / "offline_visualization.json"
    visual.write_text("{}", encoding="utf-8")
    sparse = root / "sparse_reconstruction.ply"
    sparse.write_text("ply\n", encoding="utf-8")
    payload = {
        "status": "completed",
        "pass": True,
        "session_id": sid,
        "camera_pose_available": True,
        "canonical_geometry_authoritative": False,
        "persistent_scene_ready": False,
        "certified_metrology": False,
        "artifacts": {
            "offline_visualization_json": str(visual),
            "sparse_reconstruction_ply": str(sparse),
        },
        "sfm_summary": {"gauge": {"scale_status": "arbitrary"}},
    }
    (root / "offline_result.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _admit_runtime_identity(controller: ReconstructionController, root: Path, sid: str) -> None:
    controller.offline_controller._session_id = sid
    controller.offline_controller._output_dir = root


def test_accepted_result_persists_then_forwards_one_handoff(controller, tmp_path: Path) -> None:
    root = (tmp_path / "accepted").resolve()
    root.mkdir()
    sid = str(uuid4())
    _admit_runtime_identity(controller, root, sid)
    payload = _payload(root, sid)
    handoffs = []
    forwarded = []
    controller.offline_result_handoff_ready.connect(handoffs.append)
    controller.offline_result_ready.connect(lambda result, out: forwarded.append((result, out)))

    controller._on_offline_result_ready(payload, str(root))

    assert len(handoffs) == 1
    assert handoffs[0].availability is OfflineResultHandoffAvailability.PARTIAL
    assert handoffs[0].viewer_ready is True
    assert controller.last_offline_result_handoff == handoffs[0]
    assert controller.persistence_controller.offline_result == payload
    assert forwarded == [(payload, str(root))]


def test_identity_or_provenance_rejection_is_not_forwarded(controller, tmp_path: Path) -> None:
    root = (tmp_path / "rejected").resolve()
    root.mkdir()
    sid = str(uuid4())
    _admit_runtime_identity(controller, root, sid)
    payload = _payload(root, str(uuid4()))
    errors = []
    forwarded = []
    controller.error_occurred.connect(errors.append)
    controller.offline_result_ready.connect(lambda result, out: forwarded.append((result, out)))

    controller._on_offline_result_ready(payload, str(root))

    assert forwarded == []
    assert controller.persistence_controller.offline_result is None
    assert controller.last_offline_result_handoff is None
    assert any("result handoff rejected result" in item.lower() for item in errors)


def test_new_start_state_clears_previous_handoff(controller, tmp_path: Path) -> None:
    root = (tmp_path / "previous").resolve()
    root.mkdir()
    sid = str(uuid4())
    _admit_runtime_identity(controller, root, sid)
    controller._on_offline_result_ready(_payload(root, sid), str(root))
    assert controller.last_offline_result_handoff is not None

    from app.offline_reconstruction_controller import OfflineReconstructionRunState
    controller._on_offline_state_changed(OfflineReconstructionRunState.STARTING)
    assert controller.last_offline_result_handoff is None
