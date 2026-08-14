from __future__ import annotations

from app.widgets.inspection_page import InspectionPage


def test_9_2d_measure_inspect_exposes_semantic_workspace_entry(qtbot) -> None:
    page = InspectionPage()
    qtbot.addWidget(page)
    page.show()
    qtbot.wait(5)
    buttons = [button for button in page.findChildren(type(page._capture_btn)) if button.text() == "Open Auto Segment"]
    assert len(buttons) == 1
    fired = []
    page.auto_segment_requested.connect(lambda: fired.append(True))
    buttons[0].click()
    assert fired == [True]
