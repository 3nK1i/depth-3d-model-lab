"""Spawn-safe inference worker process.

The worker owns every model and CUDA allocation. Results are latest-only: a
large geometry result is never allowed to queue behind obsolete results.
"""

from __future__ import annotations

import queue
import time
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue as MPQueue

from core.models.adapters.mock_adapter import MockGeometryModel
from core.models.adapters.moge2_adapter import Moge2GeometryModel
from core.models.base import BaseGeometryModel
from core.models.catalog import MOGE2_MODEL_ID, MOCK_MODEL_ID
from core.runtime.messages import Ack, Command, CommandType, FrameRequest, InferenceResult

_MODEL_REGISTRY: dict[str, type[BaseGeometryModel]] = {
    MOGE2_MODEL_ID: Moge2GeometryModel,
    MOCK_MODEL_ID: MockGeometryModel,
}


def _build_model(model_id: str) -> BaseGeometryModel:
    try:
        model_cls = _MODEL_REGISTRY[model_id]
    except KeyError:
        raise ValueError(f"Unknown model id: {model_id!r}") from None
    return model_cls()


def worker_main(
    control_conn: Connection,
    frame_queue: MPQueue,
    result_queue: MPQueue,
) -> None:
    """Run until a SHUTDOWN command is received."""
    active_model: BaseGeometryModel | None = None
    active_model_id: str | None = None
    paused = True

    while True:
        if control_conn.poll(timeout=0.02):
            command: Command = control_conn.recv()

            if command.type == CommandType.LOAD_MODEL:
                load_started = time.perf_counter()
                try:
                    if active_model is not None:
                        active_model.unload()
                    active_model = _build_model(command.model_id)
                    config = command.config or {}
                    active_model.load(
                        device=str(config.get("device", "cpu")),
                        precision=str(config.get("precision", "fp32")),
                        config=config,
                    )
                    active_model.warmup()
                    active_model_id = command.model_id
                    metadata = active_model.get_metadata()
                    load_time_ms = (time.perf_counter() - load_started) * 1000.0
                    control_conn.send(
                        Ack(
                            command.type,
                            True,
                            model_id=active_model_id,
                            metadata=metadata,
                            load_time_ms=load_time_ms,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - report, never crash silently
                    if active_model is not None:
                        try:
                            active_model.unload()
                        except Exception:  # noqa: BLE001
                            pass
                    active_model = None
                    active_model_id = None
                    control_conn.send(
                        Ack(
                            command.type,
                            False,
                            str(exc),
                            model_id=command.model_id,
                            load_time_ms=(time.perf_counter() - load_started) * 1000.0,
                        )
                    )

            elif command.type == CommandType.UNLOAD_MODEL:
                if active_model is not None:
                    active_model.unload()
                active_model = None
                active_model_id = None
                _drain(frame_queue)
                _drain(result_queue)
                control_conn.send(Ack(command.type, True))

            elif command.type == CommandType.PAUSE:
                paused = True
                _drain(frame_queue)
                control_conn.send(Ack(command.type, True))

            elif command.type == CommandType.RESUME:
                paused = False
                control_conn.send(Ack(command.type, True))

            elif command.type == CommandType.STOP:
                paused = True
                _drain(frame_queue)
                _drain(result_queue)
                control_conn.send(Ack(command.type, True))

            elif command.type == CommandType.SHUTDOWN:
                if active_model is not None:
                    active_model.unload()
                _drain(frame_queue)
                _drain(result_queue)
                control_conn.send(Ack(command.type, True))
                return

        if paused or active_model is None:
            continue

        try:
            frame: FrameRequest = frame_queue.get(timeout=0.02)
        except queue.Empty:
            continue

        try:
            result = active_model.infer(frame)
            _put_latest(
                result_queue,
                InferenceResult(
                    session_id=result.session_id,
                    frame_id=result.frame_id,
                    model_id=active_model_id or result.model_id,
                    depth=result.depth,
                    latency_ms=result.latency_ms,
                    points=result.points,
                    normals=result.normals,
                    intrinsics=result.intrinsics,
                    valid_mask=result.valid_mask,
                    point_stride=result.point_stride,
                    is_metric=result.is_metric,
                    model_name=result.model_name,
                    precision=result.precision,
                    vram_used_gib=result.vram_used_gib,
                    vram_total_gib=result.vram_total_gib,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - never let a bad frame kill worker
            _put_latest(
                result_queue,
                InferenceResult(
                    session_id=frame.session_id,
                    frame_id=frame.frame_id,
                    model_id=active_model_id or "",
                    depth=None,
                    latency_ms=0.0,
                    error=str(exc),
                ),
            )


def _put_latest(q: MPQueue, item: InferenceResult) -> None:
    """Replace any obsolete result before publishing the newest one."""
    _drain(q)
    try:
        q.put_nowait(item)
    except queue.Full:
        # A feeder-thread race can make ``empty then put`` briefly report Full.
        # Drop the old item one more time; never block the CUDA worker.
        _drain(q)
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


def _drain(q: MPQueue) -> None:
    """Empty a multiprocessing queue without blocking."""
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return
