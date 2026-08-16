from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from app.widgets.reconstruction_page import ReconstructionPage


def test_e3_build_processing_ui_keeps_training_action_progressively_disclosed(qtbot) -> None:
    # Let pytest-qt own QApplication lifetime. Creating a separate QApplication
    # in this test can invalidate session-scoped Qt objects (notably QSettings)
    # before pytest teardown has finished.
    page = ReconstructionPage()
    qtbot.addWidget(page)
    assert page._digital_twin_processing_group.expanded is False
    assert page._nerf_build_button.text() == "Build & Train NeRF"
    assert page._nerf_build_button.isEnabled() is False
    assert page._nerf_build_profile_combo.count() == 3
    assert page._nerf_build_profile_combo.currentData() == "efficient"
    assert page._digital_twin_processing_cancel_button.isEnabled() is False


def test_e3_persistent_scene_transition_refreshes_build_eligibility(qtbot, monkeypatch) -> None:
    page = ReconstructionPage()
    qtbot.addWidget(page)
    refresh_calls: list[bool] = []
    monkeypatch.setattr(page, "_refresh_nerf_build_controls", lambda: refresh_calls.append(True))

    # Controller attachment, scene load/reload, and explicit clear all flow through
    # set_persistent_scene(). The E3 action must be refreshed by that transition
    # itself rather than waiting for an unrelated source/runtime event.
    page.set_persistent_scene(None, "")

    assert refresh_calls == [True]
