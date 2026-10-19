from __future__ import annotations

from time import monotonic, sleep

import numpy as np
import pytest

from core.capture.frame import SourceFrame
from core.reconstruction import (
    ExternalStellaLiveSlamBackend,
    LiveCaptureRuntimeState,
    LiveCaptureSlamRuntime,
    LiveSlamRecoveryPolicy,
    STELLA_LIVE_BACKEND_ID,
)
from tests.reconstruction_live_helpers import LIVE_IMAGE_SIZE, make_live_request
from tests.reconstruction_live_worker_helpers import make_fake_stella_worker_layout


def _frame(frame_id: int) -> SourceFrame:
    return SourceFrame(
        np.zeros((LIVE_IMAGE_SIZE.height, LIVE_IMAGE_SIZE.width, 3), dtype=np.uint8),
        frame_id=frame_id,
        timestamp=monotonic() + frame_id * 0.001,
    )


def _wait_until(predicate, timeout_s: float = 3.0) -> bool:
    deadline = monotonic() + timeout_s
    while monotonic() < deadline:
        if predicate():
            return True
        sleep(0.01)
    return bool(predicate())


@pytest.mark.parametrize(
    ("mode", "frame_timeout_s"),
    [
        ("crash_frame_once", 0.4),
        ("hang_frame_once", 0.15),
        ("malformed_frame_once", 0.4),
    ],
)
def test_external_worker_crash_timeout_or_malformed_ipc_recovers_once(
    tmp_path,
    monkeypatch,
    mode: str,
    frame_timeout_s: float,
) -> None:
    marker = tmp_path / f"{mode}.marker"
    monkeypatch.setenv("DEPTH3D_FAKE_STELLA_MODE", mode)
    monkeypatch.setenv("DEPTH3D_FAKE_STELLA_MARKER", str(marker))
    prefix, worker, vocab = make_fake_stella_worker_layout(tmp_path)
    request = make_live_request(backend_id=STELLA_LIVE_BACKEND_ID)
    observations = []
    states = []
    backends = []

    def factory(ring_spec):
        backend = ExternalStellaLiveSlamBackend(
            install_prefix=prefix,
            worker_executable=worker,
            vocabulary_path=vocab,
            frame_ring_spec=ring_spec,
            frame_timeout_s=frame_timeout_s,
            shutdown_timeout_s=0.25,
            require_qualified_build=False,
        )
        backends.append(backend)
        return backend

    runtime = LiveCaptureSlamRuntime(
        request=request,
        backend_factory=factory,
        recovery_policy=LiveSlamRecoveryPolicy(max_worker_restarts=2, restart_backoff_s=0.0),
        observation_callback=observations.append,
        state_callback=states.append,
    )
    runtime.start()
    assert runtime.wait_for_state(LiveCaptureRuntimeState.RUNNING, timeout_s=3.0)
    runtime.offer_source_frame(_frame(0))
    assert _wait_until(lambda: runtime.stats.worker_restarts == 1, timeout_s=3.0)
    assert _wait_until(lambda: runtime.state is LiveCaptureRuntimeState.RUNNING, timeout_s=3.0)
    runtime.offer_source_frame(_frame(1))
    assert _wait_until(lambda: len(observations) >= 1, timeout_s=3.0)

    assert observations[-1].pose is not None
    assert runtime.stats.worker_restarts == 1
    assert runtime.stats.worker_restart_failures == 0
    assert marker.exists()
    assert LiveCaptureRuntimeState.RECOVERING in states
    assert runtime.stop(timeout_s=3.0)
    assert runtime.state is LiveCaptureRuntimeState.STOPPED
    assert runtime.stats.cleanup_complete is True
    assert all(not backend.resources_active for backend in backends)
