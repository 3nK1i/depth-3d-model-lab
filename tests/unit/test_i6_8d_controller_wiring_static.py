"""Qt-free structural guards; Qt live integration is tested in workstation UI suite."""
from pathlib import Path

OFFLINE = Path('app/offline_reconstruction_controller.py').read_text()
DENSE = Path('app/reconstruction_dense_job_controller.py').read_text()


def test_frozen_sfm_controller_is_unchanged_and_outer_controller_binds_relay():
    import hashlib
    assert hashlib.sha256(OFFLINE.encode()).hexdigest() == (
        '467b551dc2a9f183d77f49f9768363b5057ca9af4adf2f7a26b5d35c54b5b70c')
    parent = Path('app/reconstruction_controller.py').read_text()
    assert "self._offline_telemetry_relay.begin(" in parent
    assert "'offline_sfm', self._offline_controller.session_id, None," in parent
    assert 'self._offline_telemetry_relay.poll()' in parent
    assert 'self._offline_telemetry_relay.close()' in parent


def test_dense_controller_binds_run_id_and_stops_before_terminal_admission():
    assert 'telemetry_received = pyqtSignal(object)' in DENSE
    assert "self._telemetry_relay.begin('post_sparse_dense', self._session_id," in DENSE
    assert "root / 'dense_runs' / run_id" in DENSE
    assert 'self._telemetry_relay.close()' in DENSE
    assert 'self._telemetry_relay.poll()' in DENSE
    assert 'self.admission_ready.emit(self._verified_admission)' in DENSE


def test_app_reconstruction_controller_forwards_both_job_observers():
    text = Path('app/reconstruction_controller.py').read_text()
    assert 'job_telemetry_received = pyqtSignal(object)' in text
    assert 'self._offline_telemetry_relay.observation_received.connect(' in text
    assert 'self._dense_controller.telemetry_received.connect(self.job_telemetry_received.emit)' in text
