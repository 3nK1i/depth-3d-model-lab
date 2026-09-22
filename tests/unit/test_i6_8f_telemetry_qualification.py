"""I6.8F bounded telemetry transport and diagnostic probe; no reconstruction."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from scripts.run_i6_8f_telemetry_qualification import (
    _observation, host_probe, transport_probe,
)
from core.reconstruction.reconstruction_job_telemetry_contract import JobTelemetryObservation


def test_bounded_replay_history_and_factual_diagnostic_scope(tmp_path):
    result = transport_probe(tmp_path, 64)
    assert result['count'] == 64
    assert result['history_length'] == 32
    assert result['poll_median_ms'] >= 0
    assert result['poll_p95_ms'] >= 0
    assert result['poll_max_ms'] >= result['poll_p95_ms']
    assert result['tracemalloc_peak_bytes'] > 0
    assert 'no reconstruction' in result['workload']
    assert all(path.stat().st_size <= 4097 for path in tmp_path.iterdir())


def test_transport_replay_count_is_strictly_bounded(tmp_path):
    for count in (0, 31, 1025, 1.0, True):
        with pytest.raises(ValueError):
            transport_probe(tmp_path, count)


def test_synthetic_counter_is_not_an_overall_percentage_or_admission():
    event = _observation('11111111-2222-4333-8444-555555555555',
                         '66666666-7777-4888-9999-aaaaaaaaaaaa',
                         'dense_supervisor', 1)
    assert isinstance(event, JobTelemetryObservation)
    data = event.as_dict()
    assert 'progress_fraction' not in data
    assert 'eta_seconds' not in data
    assert 'final_dense_result_admitted' not in data
    assert event.counter.expected_count is None
    assert 'Synthetic qualification' in event.detail


def test_probe_does_not_start_real_reconstruction_or_modify_existing_sessions():
    source = inspect.getsource(host_probe)
    assert 'subprocess.Popen(' in source
    assert '_DIAGNOSTIC_CHILD' in source
    assert 'sampler.set_stage' in source
    assert 'child.terminate()' in source
    assert 'child.kill()' in source
    assert 'reconstruction_pycolmap_worker' not in source
    assert 'reconstruction_pycolmap_dense_provider_mvs_worker' not in source
    assert 'COLMAP' not in source
