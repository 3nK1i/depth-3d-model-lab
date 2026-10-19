from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.reconstruction_offline_workspace import (
    CANONICAL_CAMERA_AUTHORITY_OWNER,
    PRODUCTION_AUTO_POLICY_SCOPE,
    OfflineReconstructionWorkspaceSnapshot,
    OfflineWorkspaceEvidenceIdentityStatus,
    OfflineWorkspaceScaleStatus,
    build_offline_reconstruction_workspace_snapshot,
)
from app.reconstruction_session_model import build_reconstruction_session_model

SESSION = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"


def _selection(workflow: str = "image_sequence", *, ready: bool = True):
    return SimpleNamespace(
        workflow=SimpleNamespace(value=workflow),
        ready=ready,
        display_name="Factory bay capture",
        detail="24 overlapping images",
        path=Path("/tmp/i6_1a/source"),
    )


def _session(*, workflow: str = "image_sequence", ready: bool = True, offline_state: str = "idle"):
    selection = _selection(workflow, ready=ready)
    return selection, build_reconstruction_session_model(
        source_selection=selection,
        offline_session_id=SESSION if offline_state != "idle" else None,
        offline_state=SimpleNamespace(value=offline_state),
    )


def _partial_result(*, session_id: str = SESSION) -> dict[str, object]:
    return {
        "status": "completed",
        "session_id": session_id,
        "camera_pose_available": True,
        "persistent_scene_ready": False,
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
        "sfm_summary": {
            "backend_id": "pycolmap",
            "gauge": {"scale_status": "arbitrary", "unit_to_metre": None},
        },
    }


def _persistent_session():
    artifact = SimpleNamespace(
        artifact_id="artifact-camera",
        stage=SimpleNamespace(value="camera_pose"),
        kind=SimpleNamespace(value="camera_trajectory"),
        authority=SimpleNamespace(value="canonical_geometry"),
        provider_id="pycolmap_sfm",
    )
    scene = SimpleNamespace(
        scene_id="prs-i6-1a",
        session_id=SESSION,
        state=SimpleNamespace(value="ready"),
        artifacts=(artifact,),
        provider_provenance=(),
        canonical_geometry_authoritative=True,
        certified_metrology=False,
        created_at_utc="2026-09-15T08:00:00+00:00",
    )
    selection = _selection()
    return selection, build_reconstruction_session_model(
        source_selection=selection,
        scene=scene,
        scene_manifest="/tmp/i6_1a/persistent_reconstruction_scene.json",
        scene_availability=SimpleNamespace(value="ready"),
    )


def test_i6_1a_empty_workspace_is_explicit_and_claims_no_result_authority() -> None:
    selection = _selection("none", ready=False)
    session = build_reconstruction_session_model(source_selection=selection)

    snapshot = build_offline_reconstruction_workspace_snapshot(
        session_model=session,
        source_selection=selection,
    )

    assert snapshot.offline_workflow_supported is False
    assert snapshot.source_ready is False
    assert snapshot.partial_result_available is False
    assert snapshot.persistent_scene_available is False
    assert snapshot.checkpoint_available is False
    assert snapshot.camera_pose_available is False
    assert snapshot.canonical_geometry_authoritative is False
    assert snapshot.scale_status is OfflineWorkspaceScaleStatus.UNQUALIFIED
    assert snapshot.certified_metrology is False
    assert snapshot.camera_authority_owner == CANONICAL_CAMERA_AUTHORITY_OWNER
    assert snapshot.production_auto_qualified is False
    assert snapshot.production_auto_authorized is False
    assert snapshot.production_auto_policy_scope == PRODUCTION_AUTO_POLICY_SCOPE


def test_i6_1a_validated_image_sequence_is_workspace_ready_without_execution() -> None:
    selection, session = _session()

    snapshot = build_offline_reconstruction_workspace_snapshot(
        session_model=session,
        source_selection=selection,
    )

    assert snapshot.offline_workflow_supported is True
    assert snapshot.source_ready is True
    assert snapshot.workflow == "image_sequence"
    assert snapshot.lifecycle == "ready"
    assert snapshot.output_dir == ""
    assert snapshot.current_stage == ""
    assert snapshot.progress_fraction is None
    assert snapshot.result_status == ""


def test_i6_1a_recorded_video_is_supported_but_live_source_is_not_offline_ready() -> None:
    video = _selection("recorded_video", ready=True)
    video_session = build_reconstruction_session_model(source_selection=video)
    live = _selection("live_source", ready=True)
    live_session = build_reconstruction_session_model(source_selection=live)

    video_snapshot = build_offline_reconstruction_workspace_snapshot(
        session_model=video_session,
        source_selection=video,
    )
    live_snapshot = build_offline_reconstruction_workspace_snapshot(
        session_model=live_session,
        source_selection=live,
    )

    assert video_snapshot.offline_workflow_supported is True
    assert video_snapshot.source_ready is True
    assert live_snapshot.offline_workflow_supported is False
    assert live_snapshot.source_ready is False


def test_i6_1a_active_progress_is_normalized_without_owning_runtime_state() -> None:
    selection, session = _session(offline_state="running")
    event = {
        "protocol": "depth3d.reconstruction.offline.v1",
        "event": "progress",
        "stage": "camera_pose",
        "message": "Registering sparse cameras.",
        "progress_fraction": 0.625,
    }

    snapshot = build_offline_reconstruction_workspace_snapshot(
        session_model=session,
        source_selection=selection,
        offline_state=SimpleNamespace(value="running"),
        offline_output_dir="/tmp/i6_1a/output",
        latest_event=event,
    )

    assert snapshot.lifecycle == "running"
    assert snapshot.offline_run_state == "running"
    assert snapshot.current_stage == "camera_pose"
    assert snapshot.progress_fraction == pytest.approx(0.625)
    assert snapshot.progress_message == "Registering sparse cameras."
    assert snapshot.output_dir == str(Path("/tmp/i6_1a/output").resolve())


def test_i6_1a_sparse_completion_remains_partial_arbitrary_scale_and_non_authoritative() -> None:
    selection, session = _session(offline_state="completed")
    result = _partial_result()

    snapshot = build_offline_reconstruction_workspace_snapshot(
        session_model=session,
        source_selection=selection,
        offline_state=SimpleNamespace(value="completed"),
        offline_output_dir="/tmp/i6_1a/output",
        offline_result=result,
        offline_result_path="/tmp/i6_1a/output/offline_result.json",
        scene_availability=SimpleNamespace(value="partial"),
    )

    assert snapshot.partial_result_available is True
    assert snapshot.persistent_scene_available is False
    assert snapshot.camera_pose_available is True
    assert snapshot.canonical_geometry_authoritative is False
    assert snapshot.scale_status is OfflineWorkspaceScaleStatus.ARBITRARY
    assert snapshot.metric_coordinates_allowed is False
    assert snapshot.calibrated_metric_scale is False
    assert snapshot.certified_metrology is False
    assert snapshot.result_status == "completed"


def test_i6_1a_persistent_scene_and_authority_snapshot_preserve_qualified_scale_without_certification() -> None:
    selection, session = _persistent_session()
    authority = SimpleNamespace(
        metric_coordinates_allowed=True,
        calibrated_metric_scale=True,
        certified_metrology=False,
    )

    snapshot = build_offline_reconstruction_workspace_snapshot(
        session_model=session,
        source_selection=selection,
        scene_availability=SimpleNamespace(value="ready"),
        persistent_authority_snapshot=authority,
    )

    assert snapshot.persistent_scene_available is True
    assert snapshot.partial_result_available is False
    assert snapshot.camera_pose_available is True
    assert snapshot.canonical_geometry_authoritative is True
    assert snapshot.metric_coordinates_allowed is True
    assert snapshot.calibrated_metric_scale is True
    assert snapshot.scale_status is OfflineWorkspaceScaleStatus.CALIBRATED_METRIC
    assert snapshot.certified_metrology is False


def test_i6_1a_checkpoint_is_exposed_as_capability_not_resume_execution() -> None:
    checkpoint = SimpleNamespace(
        checkpoint_id="33333333-3333-4333-8333-333333333333",
        session_id=SESSION,
        next_stage=SimpleNamespace(value="dense_geometry"),
        resume_allowed=True,
        created_at_utc="2026-09-15T08:00:00+00:00",
        provider_versions=(),
        stage_results=(),
    )
    selection = _selection()
    session = build_reconstruction_session_model(
        source_selection=selection,
        checkpoint=checkpoint,
        checkpoint_manifest="/tmp/i6_1a/reconstruction_engine_checkpoint.json",
    )

    snapshot = build_offline_reconstruction_workspace_snapshot(
        session_model=session,
        source_selection=selection,
    )

    assert snapshot.checkpoint_available is True
    assert snapshot.checkpoint_resume_allowed is True
    assert snapshot.checkpoint_next_stage == "dense_geometry"
    assert snapshot.persistent_scene_available is False


def test_i6_1a_offline_result_identity_mismatch_fails_closed_as_explicit_conflict() -> None:
    selection, session = _session(offline_state="completed")

    snapshot = build_offline_reconstruction_workspace_snapshot(
        session_model=session,
        source_selection=selection,
        offline_result=_partial_result(session_id=OTHER),
    )

    assert snapshot.evidence_identity_status is OfflineWorkspaceEvidenceIdentityStatus.CONFLICT
    assert SESSION in snapshot.identity_details
    assert OTHER in snapshot.identity_details


def test_i6_1a_session_identity_conflict_is_preserved() -> None:
    selection = _selection()
    scene = SimpleNamespace(
        scene_id="prs-conflict",
        session_id=OTHER,
        state=SimpleNamespace(value="ready"),
        artifacts=(),
        provider_provenance=(),
        canonical_geometry_authoritative=True,
        certified_metrology=False,
        created_at_utc="2026-09-15T08:00:00+00:00",
    )
    session = build_reconstruction_session_model(
        source_selection=selection,
        offline_session_id=SESSION,
        offline_state=SimpleNamespace(value="running"),
        scene=scene,
        scene_availability=SimpleNamespace(value="ready"),
    )

    snapshot = build_offline_reconstruction_workspace_snapshot(
        session_model=session,
        source_selection=selection,
    )

    assert snapshot.evidence_identity_status is OfflineWorkspaceEvidenceIdentityStatus.CONFLICT
    assert snapshot.identity_conflict is True


def test_i6_1a_rejects_invalid_progress_and_forbidden_authority_promotion() -> None:
    selection, session = _session(offline_state="running")
    with pytest.raises(ValueError, match="progress_fraction"):
        build_offline_reconstruction_workspace_snapshot(
            session_model=session,
            source_selection=selection,
            latest_event={"stage": "camera_pose", "progress_fraction": 1.5},
        )

    base = build_offline_reconstruction_workspace_snapshot(
        session_model=session,
        source_selection=selection,
    )
    with pytest.raises(ValueError, match="certified metrology"):
        replace(base, certified_metrology=True)
    with pytest.raises(ValueError, match="production Auto"):
        replace(base, production_auto_authorized=True)
    with pytest.raises(ValueError, match="PyCOLMAP"):
        replace(base, camera_authority_owner="model_native_pose")
    with pytest.raises(ValueError, match="metric scale status"):
        replace(base, scale_status=OfflineWorkspaceScaleStatus.METRIC)


def test_i6_1a_rejects_result_that_attempts_certified_metrology_claim() -> None:
    selection, session = _session(offline_state="completed")
    result = _partial_result()
    result["certified_metrology"] = True

    with pytest.raises(ValueError, match="certified metrology"):
        build_offline_reconstruction_workspace_snapshot(
            session_model=session,
            source_selection=selection,
            offline_result=result,
        )


def test_i6_1a_serialization_is_stable_and_explicit_about_deferred_auto() -> None:
    selection, session = _session(offline_state="completed")
    snapshot = build_offline_reconstruction_workspace_snapshot(
        session_model=session,
        source_selection=selection,
        offline_result=_partial_result(),
        scene_availability=SimpleNamespace(value="partial"),
    )

    payload = snapshot.as_dict()
    assert payload["identity"]["session_id"] == SESSION
    assert payload["authority"]["camera_authority_owner"] == "pycolmap_sfm"
    assert payload["authority"]["certified_metrology"] is False
    assert payload["production_auto"] == {
        "policy_scope": "offline_shadow_only",
        "qualified": False,
        "authorized": False,
    }
    assert payload["result"]["partial_available"] is True


def test_i6_1a_snapshot_type_is_immutable() -> None:
    selection, session = _session()
    snapshot = build_offline_reconstruction_workspace_snapshot(
        session_model=session,
        source_selection=selection,
    )
    assert isinstance(snapshot, OfflineReconstructionWorkspaceSnapshot)
    with pytest.raises((AttributeError, TypeError)):
        snapshot.workflow = "recorded_video"  # type: ignore[misc]
