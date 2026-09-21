from __future__ import annotations

from pathlib import Path

import pytest

from app.application_controller import ApplicationController
from app.reconstruction_controller import ReconstructionController
from app.reconstruction_frame_view_plan_state import (
    FrameViewPlanIntentOrigin,
    FrameViewPlanStateStatus,
)
from app.widgets.original_view import OriginalSourceView
from app.widgets.source_bar import SourceBar
from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_frame_view_plan_contract import FrameViewPlanMode

GIB = 1024**3


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


def _images(root: Path, name: str, count: int = 80) -> Path:
    path = root / name
    path.mkdir()
    for index in range(count):
        (path / f"frame-{index:04d}.jpg").write_bytes(f"frame-{index}".encode())
    return path


def _resources(ram_gib: int = 8, free_vram_gib: int = 2) -> AdaptiveSelectionResourceSnapshot:
    return AdaptiveSelectionResourceSnapshot(
        gpu_name="NVIDIA GeForce RTX 2060",
        total_vram_bytes=6 * GIB,
        free_vram_bytes=free_vram_gib * GIB,
        available_ram_bytes=ram_gib * GIB,
        cuda_ready=True,
    )


def test_i6_5c_controller_default_is_adaptive_and_waits_for_resources(controller, tmp_path: Path) -> None:
    controller.select_image_sequence(_images(tmp_path, "a"))
    state = controller.reconstruction_frame_view_plan_state
    assert state.request.mode is FrameViewPlanMode.ADAPTIVE
    assert state.intent_origin is FrameViewPlanIntentOrigin.DEFAULT_ADAPTIVE
    assert state.status is FrameViewPlanStateStatus.WAITING_RESOURCES
    assert state.plan is None


def test_i6_5c_explicit_resources_resolve_without_starting_runtime(controller, tmp_path: Path) -> None:
    controller.select_image_sequence(_images(tmp_path, "a", 128))
    state = controller.set_reconstruction_frame_view_resource_snapshot(_resources())
    assert state.status is FrameViewPlanStateStatus.RESOLVED
    assert controller.offline_is_running is False
    assert state.plan is not None
    assert state.plan.as_dict()["effects"]["execution_started"] is False


def test_i6_5c_manual_limit_is_explicit_user_intent_only(controller, tmp_path: Path) -> None:
    controller.select_image_sequence(_images(tmp_path, "a", 120))
    controller.set_reconstruction_frame_view_resource_snapshot(_resources())
    state = controller.set_reconstruction_frame_view_plan(
        FrameViewPlanMode.MANUAL_LIMIT,
        requested_max_views=100,
    )
    assert state.intent_origin is FrameViewPlanIntentOrigin.EXPLICIT_USER_OVERRIDE
    assert state.request.requested_max_views == 100
    assert state.effective_target_views is not None
    assert state.effective_target_views <= 100
    assert controller.offline_is_running is False


def test_i6_5c_source_change_recomputes_plan_but_preserves_user_request(controller, tmp_path: Path) -> None:
    controller.select_image_sequence(_images(tmp_path, "a", 120))
    controller.set_reconstruction_frame_view_resource_snapshot(_resources())
    first = controller.set_reconstruction_frame_view_plan(
        FrameViewPlanMode.MANUAL_LIMIT,
        requested_max_views=70,
    )
    first_digest = first.planning_provenance_sha256
    first_source = first.source.source_fingerprint_sha256 if first.source else ""

    controller.select_image_sequence(_images(tmp_path, "b", 40))
    second = controller.reconstruction_frame_view_plan_state

    assert second.request.mode is FrameViewPlanMode.MANUAL_LIMIT
    assert second.request.requested_max_views == 70
    assert second.intent_origin is FrameViewPlanIntentOrigin.EXPLICIT_USER_OVERRIDE
    assert second.planning_provenance_sha256 != first_digest
    assert second.source is not None
    assert second.source.source_fingerprint_sha256 != first_source
    assert second.effective_target_views <= 40


def test_i6_5c_all_eligible_is_not_silently_capped(controller, tmp_path: Path) -> None:
    controller.select_image_sequence(_images(tmp_path, "a", 160))
    controller.set_reconstruction_frame_view_resource_snapshot(_resources(ram_gib=1, free_vram_gib=0))
    state = controller.set_reconstruction_frame_view_plan(FrameViewPlanMode.ALL_ELIGIBLE)
    assert state.status is FrameViewPlanStateStatus.RESOLVED
    assert state.effective_target_views is None
    assert state.plan is not None
    assert state.plan.resource_bound_applied is False
    assert state.plan.all_eligible_exceeds_resource_bound is True


def test_i6_5c_reset_restores_default_adaptive(controller, tmp_path: Path) -> None:
    controller.select_image_sequence(_images(tmp_path, "a"))
    controller.set_reconstruction_frame_view_resource_snapshot(_resources())
    controller.set_reconstruction_frame_view_plan(
        FrameViewPlanMode.MANUAL_LIMIT,
        requested_max_views=20,
    )
    state = controller.reset_reconstruction_frame_view_plan()
    assert state.request.mode is FrameViewPlanMode.ADAPTIVE
    assert state.request.requested_max_views is None
    assert state.intent_origin is FrameViewPlanIntentOrigin.DEFAULT_ADAPTIVE


def test_i6_5c_duplicate_updates_are_deduplicated(controller, tmp_path: Path) -> None:
    controller.select_image_sequence(_images(tmp_path, "a"))
    emitted = []
    controller.reconstruction_frame_view_plan_changed.connect(emitted.append)
    res = _resources()
    controller.set_reconstruction_frame_view_resource_snapshot(res)
    controller.set_reconstruction_frame_view_resource_snapshot(res)
    controller.set_reconstruction_frame_view_plan(FrameViewPlanMode.ALL_ELIGIBLE)
    controller.set_reconstruction_frame_view_plan(FrameViewPlanMode.ALL_ELIGIBLE)
    assert len(emitted) == 2


def test_i6_5c_invalid_manual_limit_rejected_without_runtime_effect(controller, tmp_path: Path) -> None:
    controller.select_image_sequence(_images(tmp_path, "a"))
    with pytest.raises(ReconstructionValidationError):
        controller.set_reconstruction_frame_view_plan(
            FrameViewPlanMode.MANUAL_LIMIT,
            requested_max_views=2,
        )
    assert controller.offline_is_running is False
    assert controller.reconstruction_frame_view_plan_request.mode is FrameViewPlanMode.ADAPTIVE
