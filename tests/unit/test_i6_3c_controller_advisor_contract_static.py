from pathlib import Path


CONTROLLER = Path("app/reconstruction_controller.py")


def _text() -> str:
    return CONTROLLER.read_text(encoding="utf-8")


def test_i6_3c_controller_exposes_passive_advisor_signal_and_state() -> None:
    text = _text()
    assert "reconstruction_advisor_changed = pyqtSignal(object)" in text
    assert "def reconstruction_advisor_state(self) -> ReconstructionAdvisorLiveState:" in text
    assert "build_reconstruction_advisor_live_state(" in text
    assert "self._last_emitted_reconstruction_advisor_state" in text


def test_i6_3c_explicit_binding_is_separate_from_provider_selection_or_start() -> None:
    text = _text()
    assert "def set_reconstruction_advisor_evidence_binding(" in text
    assert "def clear_reconstruction_advisor_evidence_binding(self)" in text
    assert "ReconstructionAdvisorEvidenceBinding.unresolved()" in text
    block = text[
        text.index("def set_reconstruction_advisor_evidence_binding(") :
        text.index("def scene_availability", text.index("def set_reconstruction_advisor_evidence_binding("))
    ]
    for forbidden in (
        "start_selected_offline(",
        ".start(",
        "select_provider",
        "selected_provider",
        "i5_auto_policy_engine",
    ):
        assert forbidden not in block


def test_i6_3c_live_recompute_is_deduplicated_and_passive() -> None:
    text = _text()
    assert "self.offline_workspace_changed.connect(self._emit_reconstruction_advisor_changed)" in text
    assert "self._persistence_controller.offline_result_changed.connect(" in text
    assert "if state == self._last_emitted_reconstruction_advisor_state:" in text
    assert "self.reconstruction_advisor_changed.emit(state)" in text
