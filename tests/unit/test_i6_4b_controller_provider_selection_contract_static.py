from pathlib import Path


def _controller_text() -> str:
    return Path("app/reconstruction_controller.py").read_text(encoding="utf-8")


def test_i6_4b_controller_exposes_manual_selection_state_and_signal() -> None:
    text = _controller_text()
    assert "reconstruction_provider_selection_changed = pyqtSignal(object)" in text
    assert "def reconstruction_provider_selection_state(" in text
    assert "def reconstruction_provider_selection_snapshot(" in text
    assert "def select_reconstruction_provider(" in text
    assert "def clear_reconstruction_provider_selection(" in text


def test_i6_4b_controller_does_not_wire_selection_into_execution() -> None:
    text = _controller_text()
    start = text.index("def select_reconstruction_provider(")
    end = text.index("def clear_reconstruction_provider_selection(", start)
    block = text[start:end]
    for forbidden in (
        "start_offline",
        "start_selected_offline",
        "_offline_controller.start",
        "select_dense_provider",
        "i5_auto_policy_engine",
    ):
        assert forbidden not in block
