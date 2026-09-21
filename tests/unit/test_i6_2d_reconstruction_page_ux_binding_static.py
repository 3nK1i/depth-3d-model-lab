from __future__ import annotations

from pathlib import Path


PAGE = Path("app/widgets/reconstruction_page.py")


def _source() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_page_imports_i6_2d_one_start_ux_adapter() -> None:
    source = _source()
    assert "present_offline_one_start_ux" in source


def test_page_binds_admitted_handoff_for_terminal_ux() -> None:
    source = _source()
    assert "controller.offline_result_handoff_ready.connect(self.set_offline_result_handoff)" in source
    assert "def set_offline_result_handoff(" in source


def test_new_start_clears_previous_handoff_presentation() -> None:
    source = _source()
    assert "self._offline_result_handoff = None" in source
    assert "if state is OfflineReconstructionRunState.STARTING:" in source


def test_completed_offline_state_uses_product_handoff_presentation() -> None:
    source = _source()
    assert "ux = self._offline_one_start_ux_presentation()" in source
    assert "ux.terminal_stage_label" in source
    assert "ux.terminal_message" in source


def test_runtime_controls_use_i6_2d_labels_and_tooltips_without_changing_enablement_owner() -> None:
    source = _source()
    assert "self._start_button.setText(ux.start_text)" in source
    assert "self._start_button.setToolTip(ux.start_tooltip)" in source
    assert "self._pause_button.setToolTip(ux.pause_tooltip)" in source
    assert "self._stop_button.setText(ux.stop_text)" in source
    assert "self._stop_button.setToolTip(ux.stop_tooltip)" in source
    # Existing page/controller logic must still own actual enable/disable semantics.
    assert "self._start_button.setEnabled(can_start)" in source
    assert "self._pause_button.setEnabled(can_pause)" in source
