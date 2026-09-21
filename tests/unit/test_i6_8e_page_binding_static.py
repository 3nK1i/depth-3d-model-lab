"""Guard passive Qt page telemetry wiring without requiring Qt at import time."""
from pathlib import Path

PAGE = Path('app/widgets/reconstruction_page.py').read_text()


def test_page_uses_real_controller_signal_and_active_job_identity():
    assert 'controller.job_telemetry_received.connect(self._on_job_telemetry_received)' in PAGE
    assert "'post_sparse_dense', dense.sparse_session_id" in PAGE
    assert "'offline_sfm', controller.offline_session_id" in PAGE
    assert 'self._job_telemetry_panel_state.accept(event)' in PAGE


def test_telemetry_does_not_claim_completion_or_adjust_legacy_percent_bars():
    section = PAGE.split('    def _on_job_telemetry_received(', 1)[1].split('    def set_explicit_dense_job_state(', 1)[0]
    assert '_dense_progress_bar' not in section and '_progress_bar' not in section
    assert '_dense_admission' not in section and 'setRange' not in section


def test_periodic_tick_ages_telemetry_and_terminal_states_clear_it():
    assert 'self._refresh_job_telemetry_presentation()  # stale samples never appear current' in PAGE
    assert "self._job_telemetry_panel_state.end(state.value)" in PAGE
    for name in ('reconstructionJobTelemetryStage', 'reconstructionJobTelemetryCounter',
                 'reconstructionJobTelemetryProcess', 'reconstructionJobTelemetryHost',
                 'reconstructionJobTelemetryGPU'):
        assert name in PAGE


def test_telemetry_details_do_not_compete_with_critical_runtime_buttons():
    card = PAGE.split('    def _build_runtime_card(', 1)[1].split('    def _show_job_telemetry_details(', 1)[0]
    assert 'dense_status_row.addWidget(self._job_telemetry_details_button)' in card
    assert 'layout.addWidget(self._dense_full_view_button)' in card
    assert 'self._job_telemetry_dialog = QDialog(self)' in card
    assert 'detail_layout.addWidget(label)' in card
    assert '\n        layout.addWidget(self._job_telemetry_heading)' not in card
    assert '\n        layout.addWidget(self._job_telemetry_stage)' not in card
    assert card.index('layout.addWidget(self._dense_full_view_button)') < card.index('self._job_telemetry_dialog = QDialog(self)')

def test_runtime_telemetry_disclosure_is_read_only():
    section = PAGE.split('    def _show_job_telemetry_details(', 1)[1].split('    def _build_viewport_card(', 1)[0]
    assert 'self._job_telemetry_dialog.show()' in section
    for prohibited in ('_start_runtime(', '_start_explicit_dense(', '_dense_admission', 'setRange('):
        assert prohibited not in section
