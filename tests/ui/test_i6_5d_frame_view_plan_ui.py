from __future__ import annotations

from pathlib import Path

import pytest

from app.main_window import MainWindow
from app.reconstruction_frame_view_plan_state import FrameViewPlanIntentOrigin, FrameViewPlanStateStatus
from core.reconstruction.reconstruction_frame_view_plan_contract import FrameViewPlanMode


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def _images(root: Path, name: str = "images", count: int = 8) -> Path:
    path = root / name
    path.mkdir()
    for index in range(count):
        (path / f"frame-{index:03d}.jpg").write_bytes(b"fixture")
    return path


def _settings_group_titles(page):
    layout = page._settings_scroll.widget().layout()
    titles = []
    for index in range(layout.count()):
        widget = layout.itemAt(index).widget()
        if hasattr(widget, "title"):
            titles.append(widget.title)
    return tuple(titles)


def test_i6_5d_initial_ui_is_visible_and_waits_for_source(window) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    assert controller.reconstruction_frame_view_resource_snapshot is not None
    assert page._frame_view_status_value.text() == "Waiting for source"
    assert page._frame_view_mode_combo.currentData() == FrameViewPlanMode.ADAPTIVE.value
    assert page._frame_view_manual_limit.isEnabled() is False
    assert page._frame_view_boundary_value.text().startswith("Pre-SfM intent")


def test_i6_5d_source_selection_resolves_default_adaptive_plan(window, tmp_path: Path) -> None:
    controller = window._reconstruction_controller
    controller.select_image_sequence(_images(tmp_path, count=8))
    page = window._reconstruction_page
    state = controller.reconstruction_frame_view_plan_state
    assert state.status is FrameViewPlanStateStatus.RESOLVED
    assert state.request.mode is FrameViewPlanMode.ADAPTIVE
    assert page._frame_view_status_value.text() == "Adaptive plan ready"
    assert "8 source views" == page._frame_view_source_value.text()
    assert controller.offline_is_running is False


def test_i6_5d_manual_limit_records_intent_only(window, tmp_path: Path) -> None:
    controller = window._reconstruction_controller
    controller.select_image_sequence(_images(tmp_path, count=20))
    page = window._reconstruction_page
    index = page._frame_view_mode_combo.findData(FrameViewPlanMode.MANUAL_LIMIT.value)
    page._frame_view_mode_combo.setCurrentIndex(index)
    page._on_reconstruction_frame_view_mode_changed(index)
    page._frame_view_manual_limit.setValue(7)
    page._on_reconstruction_frame_view_apply_clicked()

    state = controller.reconstruction_frame_view_plan_state
    assert state.request.mode is FrameViewPlanMode.MANUAL_LIMIT
    assert state.request.requested_max_views == 7
    assert state.intent_origin is FrameViewPlanIntentOrigin.EXPLICIT_USER_OVERRIDE
    assert controller.offline_is_running is False
    assert page._frame_view_reset_button.isEnabled() is True


def test_i6_5d_all_eligible_records_mode_without_starting(window, tmp_path: Path) -> None:
    controller = window._reconstruction_controller
    controller.select_image_sequence(_images(tmp_path, count=12))
    page = window._reconstruction_page
    index = page._frame_view_mode_combo.findData(FrameViewPlanMode.ALL_ELIGIBLE.value)
    page._frame_view_mode_combo.setCurrentIndex(index)
    page._on_reconstruction_frame_view_mode_changed(index)
    page._on_reconstruction_frame_view_apply_clicked()
    state = controller.reconstruction_frame_view_plan_state
    assert state.request.mode is FrameViewPlanMode.ALL_ELIGIBLE
    assert state.intent_origin is FrameViewPlanIntentOrigin.EXPLICIT_USER_OVERRIDE
    assert controller.offline_is_running is False


def test_i6_5d_reset_restores_default_adaptive_intent(window, tmp_path: Path) -> None:
    controller = window._reconstruction_controller
    controller.select_image_sequence(_images(tmp_path, count=12))
    controller.set_reconstruction_frame_view_plan(FrameViewPlanMode.MANUAL_LIMIT, requested_max_views=6)
    page = window._reconstruction_page
    page._on_reconstruction_frame_view_reset_clicked()
    state = controller.reconstruction_frame_view_plan_state
    assert state.request.mode is FrameViewPlanMode.ADAPTIVE
    assert state.intent_origin is FrameViewPlanIntentOrigin.DEFAULT_ADAPTIVE
    assert controller.offline_is_running is False


def test_i6_5d_source_change_preserves_request_but_recomputes_plan(window, tmp_path: Path) -> None:
    controller = window._reconstruction_controller
    controller.select_image_sequence(_images(tmp_path, "a", 20))
    controller.set_reconstruction_frame_view_plan(FrameViewPlanMode.MANUAL_LIMIT, requested_max_views=7)
    first = controller.reconstruction_frame_view_plan_state
    controller.select_image_sequence(_images(tmp_path, "b", 5))
    second = controller.reconstruction_frame_view_plan_state
    assert second.request == first.request
    assert second.plan is not None and first.plan is not None
    assert second.plan.planning_provenance_sha256 != first.plan.planning_provenance_sha256
    assert second.effective_target_views <= 5


def test_i6_5d_group_extends_frozen_i6_4d_primary_stack(window) -> None:
    titles = _settings_group_titles(window._reconstruction_page)
    assert titles[:8] == (
        "DIGITAL TWIN WORKSPACE",
        "BUILD & PROCESSING",
        "CANONICAL SCENE",
        "RULER METRIC ESTIMATE",
        "RECONSTRUCTION ADVISOR",
        "DENSE RGB POINTS",
        "PROVIDER & AUTO",
        "FRAME & VIEW PLAN",
    )
