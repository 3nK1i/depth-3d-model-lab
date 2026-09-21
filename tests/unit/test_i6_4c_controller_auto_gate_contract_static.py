from pathlib import Path


def _controller_text() -> str:
    return Path("app/reconstruction_controller.py").read_text(encoding="utf-8")


def test_i6_4c_controller_exposes_read_only_production_auto_gate() -> None:
    text = _controller_text()
    assert "def reconstruction_production_auto_gate(" in text
    assert "build_production_auto_qualification_boundary(" in text


def test_i6_4c_controller_creates_no_auto_selection_or_execution_entrypoint() -> None:
    text = _controller_text()
    for forbidden in (
        "def select_reconstruction_auto(",
        "def activate_reconstruction_auto(",
        "def start_reconstruction_auto(",
        "evaluate_i5_13b_auto_policy",
        "I513BOfflineAutoContext",
        "from core.reconstruction.i5_auto_policy_engine",
        "import core.reconstruction.i5_auto_policy_engine",
    ):
        assert forbidden not in text


def test_i6_4c_existing_manual_provider_api_remains_separate() -> None:
    text = _controller_text()
    start = text.index("def select_reconstruction_provider(")
    end = text.index("def clear_reconstruction_provider_selection(", start)
    block = text[start:end]
    assert "reconstruction_production_auto_gate" not in block
    assert "auto_policy" not in block
