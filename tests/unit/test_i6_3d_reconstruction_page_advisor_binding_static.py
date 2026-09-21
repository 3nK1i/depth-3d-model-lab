from __future__ import annotations

from pathlib import Path

PAGE = Path("app/widgets/reconstruction_page.py")


def _source() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_page_imports_advisor_state_and_presentation_adapter() -> None:
    source = _source()
    assert "ReconstructionAdvisorLiveState" in source
    assert "present_reconstruction_advisor" in source


def test_page_adds_compact_advisor_group_near_workflow() -> None:
    source = _source()
    assert 'self._settings_group("RECONSTRUCTION ADVISOR")' in source
    assert '"Advisory only • no settings changed"' in source
    assert '"Guidance"' in source
    assert '"Evidence"' in source
    assert '"Why"' in source


def test_page_binds_live_advisor_signal_and_initial_state() -> None:
    source = _source()
    assert (
        "controller.reconstruction_advisor_changed.connect("
        "self.set_reconstruction_advisor_state)"
    ) in source
    assert "self.set_reconstruction_advisor_state(controller.reconstruction_advisor_state)" in source


def test_page_advisor_binding_is_presentation_only() -> None:
    source = _source()
    start = source.index("def set_reconstruction_advisor_state(")
    end = source.index("@property\n    def reconstruction_advisor_state", start)
    body = source[start:end]
    assert "present_reconstruction_advisor(state)" in body
    assert "setText" in body
    assert "setToolTip" in body
    for forbidden in (
        "set_reconstruction_advisor_evidence_binding",
        "select_dense_provider",
        "start_selected",
        "start_offline",
        "_start_runtime",
    ):
        assert forbidden not in body


def test_advisor_group_has_no_action_button_or_provider_selector() -> None:
    source = _source()
    start = source.index('self._reconstruction_advisor_group = self._settings_group("RECONSTRUCTION ADVISOR")')
    end = source.index('session = self._settings_group("SESSION & PERSISTENCE")', start)
    block = source[start:end]
    assert "QPushButton" not in block
    assert "QComboBox" not in block


def test_advisor_group_is_promoted_into_visible_primary_sidebar_stack() -> None:
    source = _source()
    start = source.index("def _apply_digital_twin_workspace_ux_order")
    end = source.index("@staticmethod\n    def _settings_group", start)
    body = source[start:end]
    ruler = body.index("self._estimated_ruler_scale_group")
    advisor = body.index("self._reconstruction_advisor_group")
    dense = body.index("self._dense_rgb_group")
    assert ruler < advisor < dense
