"""Spawn-safe inference worker process.

The worker owns every model and CUDA allocation. Results are latest-only: a
large geometry result is never allowed to queue behind obsolete results.
"""

from __future__ import annotations

import queue
import time
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue as MPQueue
from typing import Any

from core.models.adapters.mock_adapter import MockGeometryModel
from core.models.adapters.moge2_adapter import Moge2GeometryModel
from core.models.base import BaseGeometryModel
from core.models.catalog import MOGE2_MODEL_ID, MOCK_MODEL_ID
from core.runtime.messages import (
    Ack,
    Command,
    CommandType,
    FrameRequest,
    InferenceResult,
    ModelStage,
    ModelStageEvent,
)

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
                total_started = time.perf_counter()
                model_load_time_ms: float | None = None
                warmup_time_ms: float | None = None
                try:
                    if active_model is not None:
                        _send_stage(
                            control_conn,
                            ModelStage.UNLOADING,
                            active_model_id or "",
                            "Releasing the current model before loading the new selection.",
                        )
                        active_model.unload()
                        active_model = None
                        active_model_id = None

                    _send_stage(
                        control_conn,
                        ModelStage.LOADING_WEIGHTS,
                        command.model_id,
                        "Loading model weights and allocating runtime resources.",
                    )
                    active_model = _build_model(command.model_id)
                    config = command.config or {}

                    load_started = time.perf_counter()
                    active_model.load(
                        device=str(config.get("device", "cpu")),
                        precision=str(config.get("precision", "fp32")),
                        config=config,
                    )
                    model_load_time_ms = (time.perf_counter() - load_started) * 1000.0

                    _send_stage(
                        control_conn,
                        ModelStage.WARMING_UP,
                        command.model_id,
                        "Priming model kernels and caches with a warm-up inference.",
                    )
                    warmup_started = time.perf_counter()
                    active_model.warmup()
                    warmup_time_ms = (time.perf_counter() - warmup_started) * 1000.0

                    active_model_id = command.model_id
                    metadata = active_model.get_metadata()
                    total_time_ms = (time.perf_counter() - total_started) * 1000.0
                    control_conn.send(
                        Ack(
                            command.type,
                            True,
                            model_id=active_model_id,
                            metadata=metadata,
                            load_time_ms=total_time_ms,
                            model_load_time_ms=model_load_time_ms,
                            warmup_time_ms=warmup_time_ms,
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
                            load_time_ms=(time.perf_counter() - total_started) * 1000.0,
                            model_load_time_ms=model_load_time_ms,
                            warmup_time_ms=warmup_time_ms,
                        )
                    )

            elif command.type == CommandType.UNLOAD_MODEL:
                unload_started = time.perf_counter()
                previous_model_id = active_model_id or ""
                _send_stage(
                    control_conn,
                    ModelStage.UNLOADING,
                    previous_model_id,
                    "Releasing model resources and clearing allocator caches.",
                )
                metadata: dict[str, Any] = {}
                success = True
                message = ""
                try:
                    if active_model is not None:
                        metadata = dict(active_model.unload() or {})
                except Exception as exc:  # noqa: BLE001 - acknowledge cleanup failure
                    success = False
                    message = str(exc)
                finally:
                    active_model = None
                    active_model_id = None
                    paused = True
                    _drain(frame_queue)
                    _drain(result_queue)
                control_conn.send(
                    Ack(
                        command.type,
                        success,
                        message,
                        model_id=previous_model_id,
                        metadata=metadata,
                        unload_time_ms=(time.perf_counter() - unload_started) * 1000.0,
                    )
                )

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
                success = True
                message = ""
                metadata: dict[str, Any] = {}
                unload_time_ms: float | None = None
                if active_model is not None:
                    _send_stage(
                        control_conn,
                        ModelStage.UNLOADING,
                        active_model_id or "",
                        "Releasing model resources before worker shutdown.",
                    )
                    started = time.perf_counter()
                    try:
                        metadata = dict(active_model.unload() or {})
                    except Exception as exc:  # noqa: BLE001
                        success = False
                        message = str(exc)
                    unload_time_ms = (time.perf_counter() - started) * 1000.0
                active_model = None
                active_model_id = None
                _drain(frame_queue)
                _drain(result_queue)
                control_conn.send(
                    Ack(
                        command.type,
                        success,
                        message,
                        metadata=metadata,
                        unload_time_ms=unload_time_ms,
                    )
                )
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
                    preprocess_ms=result.preprocess_ms,
                    postprocess_ms=result.postprocess_ms,
                    worker_total_ms=result.worker_total_ms,
                    queue_wait_ms=result.queue_wait_ms,
                    points=result.points,
                    normals=result.normals,
                    intrinsics=result.intrinsics,
                    valid_mask=result.valid_mask,
                    point_stride=result.point_stride,
                    is_metric=result.is_metric,
                    model_name=result.model_name,
                    precision=result.precision,
                    vram_used_gib=result.vram_used_gib,
                    vram_allocated_gib=result.vram_allocated_gib,
                    vram_reserved_gib=result.vram_reserved_gib,
                    vram_peak_reserved_gib=result.vram_peak_reserved_gib,
                    vram_device_used_gib=result.vram_device_used_gib,
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


def _send_stage(
    control_conn: Connection,
    stage: ModelStage,
    model_id: str,
    message: str,
) -> None:
    control_conn.send(ModelStageEvent(stage=stage, model_id=model_id, message=message))


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
