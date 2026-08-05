from PyQt6.QtCore import Qt

from app.stable_video_controller import StableVideoRunState
from app.widgets.stable_video_page import StableVideoPage


def test_stable_video_page_controls_and_states(qtbot):
    page = StableVideoPage()
    qtbot.addWidget(page)
    page.set_model("Video Depth Anything", native_temporal=True)
    assert "Native model cache" in page._temporal.text()
    assert page.smoothing_strength == 0.5

    page.apply_state(StableVideoRunState.RUNNING)
    assert page._start.isEnabled() is False
    assert page._pause.isEnabled() is True
    assert page._stop.isEnabled() is True

    page.apply_state(StableVideoRunState.PAUSED)
    assert page._pause.text() == "Resume"

    page.apply_state(StableVideoRunState.COMPLETED)
    assert page._start.isEnabled() is True


def test_stable_video_page_emits_stabilization(qtbot):
    page = StableVideoPage()
    qtbot.addWidget(page)
    values = []
    page.stabilization_changed.connect(values.append)
    page._stabilization.setCurrentText("High")
    assert values[-1] == 0.75
