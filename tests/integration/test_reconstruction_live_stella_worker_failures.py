from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import pytest

from core.capture.frame import SourceFrame
from core.reconstruction import (
    BackendOperationContext,
    ExternalStellaLiveSlamBackend,
    LiveFrameStaleError,
    LiveSharedMemoryFrameRing,
    LiveSlamWorkerExecutionError,
    LiveSourceFrameAdapter,
    ReconstructionCancelledError,
    ReconstructionOperationControl,
    ReconstructionStage,
    STELLA_LIVE_BACKEND_ID,
)
from tests.reconstruction_live_helpers import LIVE_IMAGE_SIZE, make_live_request
from tests.reconstruction_live_worker_helpers import make_fake_stella_worker_layout


def _setup(tmp_path, monkeypatch, mode, *, frame_timeout_s=0.4):
    monkeypatch.setenv("EIA_FAKE_STELLA_MODE", mode)
    prefix, worker, vocab = make_fake_stella_worker_layout(tmp_path)
    request = make_live_request(backend_id=STELLA_LIVE_BACKEND_ID)
    ring = LiveSharedMemoryFrameRing.create(
        slot_count=3,
        max_payload_bytes=LIVE_IMAGE_SIZE.width * LIVE_IMAGE_SIZE.height * 3,
    )
    control = ReconstructionOperationControl()
    context = BackendOperationContext(
        request.session_id,
        ReconstructionStage.LIVE_TRACKING,
        request.profile,
        control,
        lambda event: None,
    )
    backend = ExternalStellaLiveSlamBackend(
        install_prefix=prefix,
        worker_executable=worker,
        vocabulary_path=vocab,
        frame_ring_spec=ring.spec,
        frame_timeout_s=frame_timeout_s,
        require_qualified_build=False,
    )
    backend.start_live_session(request, context)
    adapter = LiveSourceFrameAdapter(
        session_id=request.session_id,
        source=request.source,
        camera=request.camera,
        ring=ring,
    )
    frame = adapter.publish(
        SourceFrame(
            image=np.zeros((LIVE_IMAGE_SIZE.height, LIVE_IMAGE_SIZE.width, 3), dtype=np.uint8),
            frame_id=0,
            timestamp=2.0,
        )
    )
    return backend, ring, context, control, frame


def test_worker_stale_generation_is_retryable_transport_error(tmp_path, monkeypatch) -> None:
    backend, ring, context, _, frame = _setup(tmp_path, monkeypatch, "stale_frame")
    try:
        with pytest.raises(LiveFrameStaleError, match="overwritten"):
            backend.track_live_frame(frame, context)
    finally:
        backend.shutdown()
        ring.shutdown()


def test_worker_crash_is_detected_as_typed_execution_failure(tmp_path, monkeypatch) -> None:
    backend, ring, context, _, frame = _setup(tmp_path, monkeypatch, "crash_frame")
    try:
        with pytest.raises(LiveSlamWorkerExecutionError, match="exited unexpectedly|closed"):
            backend.track_live_frame(frame, context)
    finally:
        backend.shutdown()
        ring.shutdown()


def test_worker_frame_timeout_is_bounded(tmp_path, monkeypatch) -> None:
    backend, ring, context, _, frame = _setup(
        tmp_path, monkeypatch, "hang_frame", frame_timeout_s=0.2
    )
    try:
        with pytest.raises(LiveSlamWorkerExecutionError, match="Timed out"):
            backend.track_live_frame(frame, context)
    finally:
        backend.shutdown()
        ring.shutdown()


def test_worker_wait_observes_cooperative_cancellation(tmp_path, monkeypatch) -> None:
    backend, ring, context, control, frame = _setup(
        tmp_path, monkeypatch, "cancel_frame", frame_timeout_s=5.0
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(backend.track_live_frame, frame, context)
            time.sleep(0.15)
            control.request_cancel()
            with pytest.raises(ReconstructionCancelledError):
                future.result(timeout=2.0)
    finally:
        backend.shutdown()
        ring.shutdown()
