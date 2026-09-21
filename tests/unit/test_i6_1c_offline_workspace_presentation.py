from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from app.reconstruction_offline_workspace import (
    OfflineWorkspaceEvidenceIdentityStatus,
    OfflineWorkspaceScaleStatus,
    build_offline_reconstruction_workspace_snapshot,
)
from app.reconstruction_offline_workspace_presentation import (
    present_offline_reconstruction_workspace,
)
from app.reconstruction_session_model import build_reconstruction_session_model


def _snapshot(**changes):
    selection = SimpleNamespace(
        workflow=SimpleNamespace(value="none"),
        ready=False,
        display_name="",
        detail="",
        path=None,
    )
    session = build_reconstruction_session_model(source_selection=selection)
    base = build_offline_reconstruction_workspace_snapshot(
        session_model=session,
        source_selection=selection,
    )
    return replace(base, **changes)


def test_i6_1c_idle_workspace_is_compact_and_neutral() -> None:
    presentation = present_offline_reconstruction_workspace(_snapshot())

    assert presentation.workspace_text == "Offline idle"
    assert presentation.workspace_status_kind == "neutral"
    assert presentation.authority_text == "PyCOLMAP • scale unqualified"
    assert presentation.authority_status_kind == "neutral"


def test_i6_1c_ready_source_is_visibly_ready_without_execution() -> None:
    presentation = present_offline_reconstruction_workspace(
        _snapshot(
            workflow="image_sequence",
            offline_workflow_supported=True,
            source_ready=True,
            lifecycle="ready",
        )
    )

    assert presentation.workspace_text == "Ready to reconstruct"
    assert presentation.workspace_status_kind == "ready"
    assert "image sequence" in presentation.workspace_tooltip.lower()


def test_i6_1c_running_workspace_surfaces_stage_without_replacing_runtime_status() -> None:
    presentation = present_offline_reconstruction_workspace(
        _snapshot(
            workflow="recorded_video",
            offline_workflow_supported=True,
            source_ready=True,
            lifecycle="running",
            offline_run_state="running",
            current_stage="camera_pose",
            progress_fraction=0.42,
            progress_message="Registering sparse cameras.",
        )
    )

    assert presentation.workspace_text == "Reconstructing"
    assert presentation.workspace_status_kind == "running"
    assert "Camera pose" in presentation.workspace_tooltip
    assert "42%" in presentation.workspace_tooltip


def test_i6_1c_partial_sparse_result_is_not_presented_as_final_scene() -> None:
    presentation = present_offline_reconstruction_workspace(
        _snapshot(
            workflow="image_sequence",
            offline_workflow_supported=True,
            source_ready=True,
            offline_run_state="completed",
            result_status="completed",
            scene_availability="partial",
            partial_result_available=True,
            camera_pose_available=True,
            scale_status=OfflineWorkspaceScaleStatus.ARBITRARY,
        )
    )

    assert presentation.workspace_text == "Partial result"
    assert presentation.workspace_status_kind == "warning"
    assert "not a persistent canonical scene" in presentation.workspace_tooltip
    assert presentation.authority_text == "PyCOLMAP • arbitrary scale"
    assert presentation.authority_status_kind == "warning"


def test_i6_1c_persistent_metric_scene_is_ready_but_not_certified_metrology() -> None:
    presentation = present_offline_reconstruction_workspace(
        _snapshot(
            workflow="image_sequence",
            offline_workflow_supported=True,
            source_ready=True,
            scene_availability="ready",
            persistent_scene_available=True,
            canonical_geometry_authoritative=True,
            camera_pose_available=True,
            metric_coordinates_allowed=True,
            scale_status=OfflineWorkspaceScaleStatus.METRIC,
        )
    )

    assert presentation.workspace_text == "Scene ready"
    assert presentation.workspace_status_kind == "ready"
    assert presentation.authority_text == "PyCOLMAP • metric coordinates"
    assert presentation.authority_status_kind == "ready"
    assert "Certified metrology: no" in presentation.authority_tooltip


def test_i6_1c_identity_conflict_is_fail_closed_and_high_visibility() -> None:
    presentation = present_offline_reconstruction_workspace(
        _snapshot(
            evidence_identity_status=OfflineWorkspaceEvidenceIdentityStatus.CONFLICT,
            identity_details="session=A • offline_result=B",
        )
    )

    assert presentation.workspace_text == "Evidence conflict"
    assert presentation.workspace_status_kind == "error"
    assert "session=A" in presentation.workspace_tooltip


def test_i6_1c_authority_tooltip_keeps_production_auto_deferred() -> None:
    presentation = present_offline_reconstruction_workspace(_snapshot())

    assert "Production Auto: deferred" in presentation.authority_tooltip
    assert "offline/shadow" in presentation.authority_tooltip
