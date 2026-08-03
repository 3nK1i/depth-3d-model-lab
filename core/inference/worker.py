"""Spawn-safe inference worker process.

The worker owns every model and CUDA allocation. Results are latest-only: a
large geometry result is never allowed to queue behind obsolete results.
Structured failures are returned to the UI instead of leaking raw exception
strings across the process boundary.
"""

from __future__ import annotations

import logging
import queue
import time
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue as MPQueue
from typing import Any

from core.models.adapters.mock_adapter import MockGeometryModel
from core.models.adapters.moge2_adapter import Moge2GeometryModel
from core.models.base import BaseGeometryModel
from core.models.catalog import MOGE2_MODEL_ID, MOCK_MODEL_ID
from core.runtime.failures import FailureStage, classify_failure
from core.runtime.logging_setup import configure_logging
from core.runtime.messages import (
    Ack,
    Command,
    CommandType,
    FrameRequest,
    InferenceResult,
    ModelStage,
    ModelStageEvent,
)

logger = logging.getLogger(__name__)

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
    try:
        log_path = configure_logging(process_name="worker", console=False)
        logger.info("Inference worker started; log=%s", log_path)
    except Exception:  # noqa: BLE001 - logging must never block inference startup
        pass

    active_model: BaseGeometryModel | None = None
    active_model_id: str | None = None
    paused = True

    while True:
        try:
            has_command = control_conn.poll(timeout=0.02)
        except (EOFError, OSError):
            logger.info("Parent control channel closed; stopping worker.")
            return

        if has_command:
            try:
                command: Command = control_conn.recv()
            except (EOFError, OSError):
                logger.info("Parent control channel closed during receive; stopping worker.")
                return

            if command.type == CommandType.LOAD_MODEL:
                total_started = time.perf_counter()
                model_load_time_ms: float | None = None
                warmup_time_ms: float | None = None
                failure_stage = FailureStage.LOAD
                cleanup_metadata: dict[str, Any] = {}
                try:
                    if active_model is not None:
                        failure_stage = FailureStage.UNLOAD
                        _send_stage(
                            control_conn,
                            ModelStage.UNLOADING,
                            active_model_id or "",
                            "Releasing the current model before loading the new selection.",
                        )
                        cleanup_metadata = dict(active_model.unload() or {})
                        active_model = None
                        active_model_id = None

                    failure_stage = FailureStage.LOAD
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

                    failure_stage = FailureStage.WARMUP
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
                    metadata = dict(active_model.get_metadata() or {})
                    if cleanup_metadata:
                        metadata["previous_model_cleanup"] = cleanup_metadata
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
                    logger.exception("Model load/warm-up failed for %s", command.model_id)
                    failure = classify_failure(
                        exc,
                        stage=failure_stage,
                        model_id=command.model_id,
                    )
                    if active_model is not None:
                        try:
                            cleanup_metadata = dict(active_model.unload() or {})
                        except Exception:  # noqa: BLE001 - preserve original failure
                            pass
                    active_model = None
                    active_model_id = None
                    paused = True
                    _drain(frame_queue)
                    _drain(result_queue)
                    control_conn.send(
                        Ack(
                            command.type,
                            False,
                            str(exc),
                            model_id=command.model_id,
                            metadata=cleanup_metadata or None,
                            load_time_ms=(time.perf_counter() - total_started) * 1000.0,
                            model_load_time_ms=model_load_time_ms,
                            warmup_time_ms=warmup_time_ms,
                            failure=failure,
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
                failure = None
                message = ""
                try:
                    if active_model is not None:
                        metadata = dict(active_model.unload() or {})
                except Exception as exc:  # noqa: BLE001 - acknowledge cleanup failure
                    logger.exception("Model unload failed for %s", previous_model_id)
                    message = str(exc)
                    failure = classify_failure(
                        exc,
                        stage=FailureStage.UNLOAD,
                        model_id=previous_model_id,
                    )
                finally:
                    active_model = None
                    active_model_id = None
                    paused = True
                    _drain(frame_queue)
                    _drain(result_queue)
                control_conn.send(
                    Ack(
                        command.type,
                        failure is None,
                        message,
                        model_id=previous_model_id,
                        metadata=metadata,
                        unload_time_ms=(time.perf_counter() - unload_started) * 1000.0,
                        failure=failure,
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
                message = ""
                metadata: dict[str, Any] = {}
                failure = None
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
                        logger.exception("Model cleanup failed during worker shutdown")
                        message = str(exc)
                        failure = classify_failure(
                            exc,
                            stage=FailureStage.UNLOAD,
                            model_id=active_model_id or "",
                        )
                    unload_time_ms = (time.perf_counter() - started) * 1000.0
                active_model = None
                active_model_id = None
                _drain(frame_queue)
                _drain(result_queue)
                control_conn.send(
                    Ack(
                        command.type,
                        failure is None,
                        message,
                        metadata=metadata,
                        unload_time_ms=unload_time_ms,
                        failure=failure,
                    )
                )
                logger.info("Inference worker shutdown complete")
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
            failed_model_id = active_model_id or ""
            logger.exception("Inference failed for model %s", failed_model_id)
            failure = classify_failure(
                exc,
                stage=FailureStage.INFERENCE,
                model_id=failed_model_id,
            )
            paused = True
            _drain(frame_queue)
            cleanup_metadata: dict[str, Any] = {}
            if failure.requires_worker_restart and active_model is not None:
                try:
                    cleanup_metadata = dict(active_model.unload() or {})
                except Exception:  # noqa: BLE001 - preserve inference failure
                    pass
                active_model = None
                active_model_id = None
                _drain(result_queue)
            _put_latest(
                result_queue,
                InferenceResult(
                    session_id=frame.session_id,
                    frame_id=frame.frame_id,
                    model_id=failed_model_id,
                    depth=None,
                    latency_ms=0.0,
                    error=str(exc),
                    failure=failure,
                    vram_allocated_gib=_optional_float(cleanup_metadata, "after_allocated_gib"),
                    vram_reserved_gib=_optional_float(cleanup_metadata, "after_reserved_gib"),
                    vram_device_used_gib=_optional_float(cleanup_metadata, "after_device_used_gib"),
                    vram_total_gib=_optional_float(cleanup_metadata, "total_gib"),
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


def _optional_float(metadata: dict[str, Any], key: str) -> float | None:
    value = metadata.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None
