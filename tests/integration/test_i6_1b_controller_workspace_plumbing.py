from __future__ import annotations

from pathlib import Path

import pytest

from app.application_controller import ApplicationController
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.reconstruction_controller import ReconstructionController
from app.reconstruction_offline_workspace import (
    OfflineWorkspaceEvidenceIdentityStatus,
    OfflineWorkspaceScaleStatus,
)
from app.widgets.original_view import OriginalSourceView
from app.widgets.source_bar import SourceBar

SESSION = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"


@pytest.fixture
def application_controller(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    source_bar = SourceBar()
    original_view = OriginalSourceView()
    qtbot.addWidget(source_bar)
    qtbot.addWidget(original_view)
    controller = ApplicationController(source_bar, original_view)
    yield controller
    controller.shutdown()


def _image_sequence(tmp_path: Path) -> Path:
    root = tmp_path / "i6-1b-images"
    root.mkdir()
    for index in range(3):
        (root / f"frame-{index:02d}.jpg").write_bytes(b"fixture")
    return root


def test_i6_1b_initial_workspace_snapshot_is_passive_and_preserves_authority(
    application_controller,
) -> None:
    controller = ReconstructionController(application_controller)

    snapshot = controller.offline_workspace_snapshot

    assert snapshot.workflow == "none"
    assert snapshot.source_ready is False
    assert snapshot.offline_workflow_supported is False
    assert snapshot.camera_authority_owner == "pycolmap_sfm"
    assert snapshot.scale_status is OfflineWorkspaceScaleStatus.UNQUALIFIED
    assert snapshot.production_auto_policy_scope == "offline_shadow_only"
    assert snapshot.production_auto_qualified is False
    assert snapshot.production_auto_authorized is False

    controller.shutdown()


def test_i6_1b_source_selection_recomposes_workspace_and_emits_once(
    application_controller, tmp_path: Path
) -> None:
    controller = ReconstructionController(application_controller)
    emitted = []
    controller.offline_workspace_changed.connect(emitted.append)

    images = _image_sequence(tmp_path)
    controller.select_image_sequence(images)

    assert len(emitted) == 1
    snapshot = emitted[-1]
    assert snapshot == controller.offline_workspace_snapshot
    assert snapshot.workflow == "image_sequence"
    assert snapshot.offline_workflow_supported is True
    assert snapshot.source_ready is True

    controller.shutdown()


def test_i6_1b_offline_event_is_forwarded_cached_and_deduplicated(
    application_controller, tmp_path: Path
) -> None:
    controller = ReconstructionController(application_controller)
    controller.select_image_sequence(_image_sequence(tmp_path))
    forwarded = []
    workspaces = []
    controller.offline_event_received.connect(forwarded.append)
    controller.offline_workspace_changed.connect(workspaces.append)

    event = {
        "protocol": "depth3d.reconstruction.offline.v1",
        "event": "progress",
        "stage": "camera_pose",
        "message": "Sparse reconstruction running.",
        "progress_fraction": 0.4,
    }
    controller.offline_controller.event_received.emit(event)
    controller.offline_controller.event_received.emit(dict(event))

    assert forwarded == [event, event]
    assert len(workspaces) == 1
    snapshot = controller.offline_workspace_snapshot
    assert snapshot.current_stage == "camera_pose"
    assert snapshot.progress_fraction == pytest.approx(0.4)
    assert snapshot.progress_message == "Sparse reconstruction running."

    controller.shutdown()


def test_i6_1b_starting_state_clears_stale_progress_before_recomposition(
    application_controller, tmp_path: Path
) -> None:
    controller = ReconstructionController(application_controller)
    controller.select_image_sequence(_image_sequence(tmp_path))
    controller.offline_controller.event_received.emit(
        {
            "protocol": "depth3d.reconstruction.offline.v1",
            "event": "progress",
            "stage": "camera_pose",
            "message": "Old run progress",
            "progress_fraction": 0.8,
        }
    )

    controller.offline_controller._state = OfflineReconstructionRunState.STARTING
    controller.offline_controller.state_changed.emit(OfflineReconstructionRunState.STARTING)

    snapshot = controller.offline_workspace_snapshot
    assert snapshot.offline_run_state == "starting"
    assert snapshot.current_stage == ""
    assert snapshot.progress_fraction is None
    assert snapshot.progress_message == ""

    controller.shutdown()


def test_i6_1b_persistence_result_recomposes_partial_workspace_without_promotion(
    application_controller, tmp_path: Path
) -> None:
    controller = ReconstructionController(application_controller)
    controller.select_image_sequence(_image_sequence(tmp_path))
    controller.offline_controller._session_id = SESSION
    emitted = []
    controller.offline_workspace_changed.connect(emitted.append)

    output_dir = tmp_path / "offline-result"
    output_dir.mkdir()
    controller.persistence_controller.accept_offline_result(
        {
            "session_id": SESSION,
            "status": "completed",
            "pass": True,
            "camera_pose_available": True,
            "persistent_scene_ready": False,
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
            "sfm_summary": {"gauge": {"scale_status": "arbitrary"}},
        },
        output_dir,
    )

    snapshot = controller.offline_workspace_snapshot
    assert emitted
    assert snapshot.evidence_identity_status is OfflineWorkspaceEvidenceIdentityStatus.CONSISTENT
    assert snapshot.partial_result_available is True
    assert snapshot.persistent_scene_available is False
    assert snapshot.result_status == "completed"
    assert snapshot.result_path == str((output_dir / "offline_result.json").resolve())
    assert snapshot.scale_status is OfflineWorkspaceScaleStatus.ARBITRARY
    assert snapshot.canonical_geometry_authoritative is False

    controller.shutdown()


def test_i6_1b_result_identity_conflict_is_visible_not_merged(
    application_controller, tmp_path: Path
) -> None:
    controller = ReconstructionController(application_controller)
    controller.select_image_sequence(_image_sequence(tmp_path))
    controller.offline_controller._session_id = SESSION

    output_dir = tmp_path / "conflicting-result"
    output_dir.mkdir()
    controller.persistence_controller.accept_offline_result(
        {
            "session_id": OTHER,
            "status": "completed",
            "pass": True,
            "camera_pose_available": True,
            "persistent_scene_ready": False,
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        },
        output_dir,
    )

    snapshot = controller.offline_workspace_snapshot
    assert snapshot.evidence_identity_status is OfflineWorkspaceEvidenceIdentityStatus.CONFLICT
    assert snapshot.identity_conflict is True
    assert SESSION in snapshot.identity_details
    assert OTHER in snapshot.identity_details

    controller.shutdown()


def test_i6_1b_authority_change_recomposes_scale_without_auto_or_ui_side_effects(
    application_controller,
) -> None:
    controller = ReconstructionController(application_controller)
    emitted = []
    controller.offline_workspace_changed.connect(emitted.append)

    class Authority:
        certified_metrology = False
        metric_coordinates_allowed = True
        calibrated_metric_scale = False

    controller.persistence_controller._authority_snapshot = Authority()
    controller.persistence_controller.persistent_authority_changed.emit(Authority(), "")

    assert emitted
    snapshot = controller.offline_workspace_snapshot
    assert snapshot.metric_coordinates_allowed is True
    assert snapshot.scale_status is OfflineWorkspaceScaleStatus.METRIC
    assert snapshot.certified_metrology is False
    assert snapshot.production_auto_qualified is False
    assert snapshot.production_auto_authorized is False

    controller.shutdown()
