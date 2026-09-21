from __future__ import annotations

from pathlib import Path


PAGE = Path("app/widgets/reconstruction_page.py")


def _source() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_i6_1c_page_imports_workspace_contract_and_presentation() -> None:
    source = _source()
    assert "OfflineReconstructionWorkspaceSnapshot" in source
    assert "present_offline_reconstruction_workspace" in source


def test_i6_1c_page_exposes_compact_workspace_and_authority_rows() -> None:
    source = _source()
    assert 'self._add_status_row(layout, "Workspace", self._workspace_value)' in source
    assert 'self._add_status_row(layout, "Authority", self._workspace_authority_value)' in source


def test_i6_1c_page_binds_controller_workspace_signal_and_initial_snapshot() -> None:
    source = _source()
    assert "controller.offline_workspace_changed.connect(self.set_offline_workspace_snapshot)" in source
    assert "self.set_offline_workspace_snapshot(controller.offline_workspace_snapshot)" in source


def test_i6_1c_workspace_setter_is_presentation_only() -> None:
    source = _source()
    assert "def set_offline_workspace_snapshot(" in source
    assert "self._workspace_value.setText(presentation.workspace_text)" in source
    assert "self._workspace_authority_value.setText(presentation.authority_text)" in source
    # UI binding must not activate provider Auto or execute reconstruction.
    assert "select_only_on_exact_unique_evidence_support_else_abstain" not in source
