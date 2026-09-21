from __future__ import annotations

from pathlib import Path


def test_i6_5d_page_wires_frame_view_state_and_controls_without_execution() -> None:
    text = Path("app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    for required in (
        'self._frame_view_plan_group = self._settings_group("FRAME & VIEW PLAN")',
        'controller.reconstruction_frame_view_plan_changed.connect(',
        'self.set_reconstruction_frame_view_plan_state',
        'controller.set_reconstruction_frame_view_plan(',
        'controller.reset_reconstruction_frame_view_plan()',
        'reconstructionFrameViewModeCombo',
        'reconstructionFrameViewManualLimit',
    ):
        assert required in text

    start = text.index("def _on_reconstruction_frame_view_apply_clicked(")
    end = text.index("def _on_reconstruction_frame_view_reset_clicked(", start)
    block = text[start:end]
    for forbidden in (
        "start_offline",
        "start_selected_offline",
        "_offline_controller.start",
        "select_dense_provider",
        "pycolmap",
    ):
        assert forbidden not in block


def test_i6_5d_main_window_supplies_resources_outside_controller() -> None:
    text = Path("app/main_window.py").read_text(encoding="utf-8")
    assert "build_startup_frame_view_resource_snapshot" in text
    assert "set_reconstruction_frame_view_resource_snapshot" in text
