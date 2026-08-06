from __future__ import annotations

from PyQt6.QtCore import Qt
from app.widgets.inspection_page import InspectionPage
from core.inspection import load_inspection_bundle
from tests.inspection_helpers import write_inspection_bundle


def test_export_controls_follow_bundle_state(qtbot, tmp_path) -> None:
    page = InspectionPage()
    qtbot.addWidget(page)
    assert not page._export_selected_btn.isEnabled()
    assert not page._export_all_btn.isEnabled()
    summary, arrays = load_inspection_bundle(write_inspection_bundle(tmp_path / "bundle"))
    page.load_bundle(summary, arrays)
    assert page._export_selected_btn.isEnabled()
    assert page._export_all_btn.isEnabled()
    context = page.export_context("selected")
    assert context["observation_ids"] == ("object-0",)


def test_export_buttons_emit_scope(qtbot) -> None:
    page = InspectionPage()
    qtbot.addWidget(page)
    emitted = []
    page.export_requested.connect(emitted.append)
    page._export_selected_btn.setEnabled(True)
    page._export_all_btn.setEnabled(True)
    qtbot.mouseClick(page._export_selected_btn, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(page._export_all_btn, Qt.MouseButton.LeftButton)
    assert emitted == ["selected", "all"]
