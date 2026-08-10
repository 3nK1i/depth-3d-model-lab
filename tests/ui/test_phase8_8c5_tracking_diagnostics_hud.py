from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtCore import Qt

from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from core.reconstruction import LiveCaptureRuntimeState


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    win._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    yield win
    win.close()


def _observation(
    *,
    state: str = "tracking",
    latency_ms: float = 14.8,
    revision: int = 9,
    keyframes: int = 9,
    landmarks: int = 547,
):
    pose = np.eye(4, dtype=np.float64)
    pose[2, 3] = 0.3
    return SimpleNamespace(
        tracking_state=SimpleNamespace(value=state),
        quality=SimpleNamespace(
            processing_latency_ns=int(latency_ms * 1_000_000),
            tracked_features=0,
            inlier_features=0,
            inlier_ratio=None,
            reason="stella_pose_valid" if state == "tracking" else f"stella_{state}",
        ),
        pose=SimpleNamespace(T_reconstruction_from_camera=pose),
        is_keyframe=False,
        map_snapshot=SimpleNamespace(
            revision=revision,
            keyframe_count=keyframes,
            landmark_count=landmarks,
        ),
    )


def test_c5_diagnostics_hud_is_top_level_viewport_overlay_with_explicit_toggle(window, qtbot) -> None:
    page = window._reconstruction_page
    button = page._viewport_tool_buttons["diagnostics"]
    assert button.isChecked() is True
    assert page._diagnostics_hud.isHidden() is True
    page.set_live_state(LiveCaptureRuntimeState.RUNNING)
    assert page._diagnostics_hud.isHidden() is False
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert page._diagnostics_hud.isHidden() is True
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert page._diagnostics_hud.isHidden() is False


def test_c5_live_observation_updates_tracking_latency_and_map_diagnostics(window, qtbot) -> None:
    page = window._reconstruction_page
    page.set_live_state(LiveCaptureRuntimeState.RUNNING)
    page.set_live_observation(_observation())
    qtbot.wait(55)

    presentation = page._diagnostics_hud.presentation
    assert presentation.tracking_state == "tracking"
    assert page._diagnostics_hud._tracking_badge.text() == "TRACKING"
    assert page._diagnostics_hud._latency_value.text().startswith("14.8 ms")
    assert page._diagnostics_hud._map_value.text() == "r9  •  9 KF  •  547 LM"
    assert page._diagnostics_hud._features_value.text() == "Not reported by Stella"
    assert presentation.feature_quality_available is False


def test_c5_map_growth_is_revision_based_not_fabricated_quality_score(window) -> None:
    page = window._reconstruction_page
    page.set_live_state(LiveCaptureRuntimeState.RUNNING)
    page.set_live_observation(_observation(revision=9, keyframes=9, landmarks=547))
    page.set_live_observation(_observation(revision=10, keyframes=10, landmarks=612))
    presentation = page._diagnostics_hud.presentation
    assert presentation.map_growth_text == "+1 KF  •  +65 LM"
    assert "quality" not in page._diagnostics_hud._growth_value.text().lower()


def test_c5_terminal_state_retains_last_diagnostics_and_labels_it_as_last(window) -> None:
    page = window._reconstruction_page
    page.set_live_state(LiveCaptureRuntimeState.RUNNING)
    page.set_live_observation(_observation())
    page.set_live_state(LiveCaptureRuntimeState.STOPPED)

    presentation = page._diagnostics_hud.presentation
    assert presentation.terminal is True
    assert page._diagnostics_hud._tracking_badge.text() == "LAST • TRACKING"
    assert page._diagnostics_hud._map_value.text() == "r9  •  9 KF  •  547 LM"


def test_c5_new_session_clears_diagnostics_but_preserves_user_visibility_preference(window, qtbot) -> None:
    page = window._reconstruction_page
    page.set_live_state(LiveCaptureRuntimeState.RUNNING)
    page.set_live_observation(_observation())
    assert page._live_diagnostics.latency_sample_count == 1

    button = page._viewport_tool_buttons["diagnostics"]
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert button.isChecked() is False
    page.set_live_state(LiveCaptureRuntimeState.STOPPED)
    page.set_live_state(LiveCaptureRuntimeState.STARTING)

    assert page._live_diagnostics.latency_sample_count == 0
    assert page._diagnostics_hud.presentation.tracking_state == "unknown"
    assert page._diagnostics_hud._tracking_badge.text() == "WAITING"
    assert button.isChecked() is False
    assert page._diagnostics_hud.isHidden() is True
