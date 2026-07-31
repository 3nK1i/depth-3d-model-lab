"""Inference worker process entry point (spec §12, §13).

Runs in a separate ``multiprocessing`` process, never a thread — a
thread cannot be safely force-stopped if a CUDA call hangs, but a
process can be terminated by the OS as a last resort (the "Restart
Worker" path). This module deliberately knows nothing about sessions
being "active" — that check happens entirely on the UI side, per spec
§13 ("the UI must ignore results when result.session_id !=
active_session_id"). The worker just processes whatever frame arrives
and tags the result with the same ids it received; staleness is the
caller's problem to filter, not the worker's to track.

STOP does not unload the model (spec §13: "keep the model loaded for
fast restart") — only UNLOAD_MODEL and SHUTDOWN release it.
"""

from __future__ import annotations

import queue
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue as MPQueue

from core.models.adapters.mock_adapter import MockGeometryModel
from core.models.base import BaseGeometryModel
from core.runtime.messages import Ack, Command, CommandType, FrameRequest, InferenceResult

# Registry of buildable model adapters by id. Real adapters (MoGe-2,
# Depth Anything V2, ...) register here starting in Phase 5; the worker
# dispatch logic above never needs to change when they're added.
_MODEL_REGISTRY: dict[str, type[BaseGeometryModel]] = {
    "development-mock": MockGeometryModel,
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
    """The worker process's whole lifetime. Returns (process exits)
    only after a SHUTDOWN command."""
    active_model: BaseGeometryModel | None = None
    active_model_id: str | None = None
    paused = True  # nothing runs until a LOAD_MODEL + RESUME/frames arrive

    while True:
        if control_conn.poll(timeout=0.02):
            command: Command = control_conn.recv()

            if command.type == CommandType.LOAD_MODEL:
                try:
                    if active_model is not None:
                        active_model.unload()
                    active_model = _build_model(command.model_id)
                    config = command.config or {}
                    active_model.load(
                        device=config.get("device", "cpu"),
                        precision=config.get("precision", "fp32"),
                        config=config,
                    )
                    active_model.warmup()
                    active_model_id = command.model_id
                    control_conn.send(Ack(command.type, True))
                except Exception as exc:  # noqa: BLE001 - report any load failure, never crash silently
                    active_model = None
                    active_model_id = None
                    control_conn.send(Ack(command.type, False, str(exc)))

            elif command.type == CommandType.UNLOAD_MODEL:
                if active_model is not None:
                    active_model.unload()
                active_model = None
                active_model_id = None
                control_conn.send(Ack(command.type, True))

            elif command.type == CommandType.PAUSE:
                paused = True
                control_conn.send(Ack(command.type, True))

            elif command.type == CommandType.RESUME:
                paused = False
                control_conn.send(Ack(command.type, True))

            elif command.type == CommandType.STOP:
                paused = True
                _drain(frame_queue)
                control_conn.send(Ack(command.type, True))

            elif command.type == CommandType.SHUTDOWN:
                if active_model is not None:
                    active_model.unload()
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
            result_queue.put(
                InferenceResult(
                    session_id=result.session_id,
                    frame_id=result.frame_id,
                    model_id=active_model_id or "",
                    depth=result.depth,
                    latency_ms=result.latency_ms,
                    is_metric=result.is_metric,
                )
            )
        except Exception as exc:  # noqa: BLE001 - never let a bad frame kill the worker
            result_queue.put(
                InferenceResult(
                    session_id=frame.session_id,
                    frame_id=frame.frame_id,
                    model_id=active_model_id or "",
                    depth=None,
                    latency_ms=0.0,
                    error=str(exc),
                )
            )


def _drain(q: MPQueue) -> None:
    """Empty a multiprocessing queue without blocking."""
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return