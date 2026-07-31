"""Picklable IPC message types between the UI process and the inference
worker process.

Every message crossing the process boundary (via the control Pipe or
the frame/result Queues) must be picklable — no Qt objects, no live
CUDA handles, no open file descriptors. Plain dataclasses of primitives
and numpy arrays only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class CommandType(Enum):
    LOAD_MODEL = "load_model"
    UNLOAD_MODEL = "unload_model"
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True, slots=True)
class Command:
    """Sent from the UI process to the worker over the control pipe."""

    type: CommandType
    session_id: int = 0
    model_id: str = ""
    # Free-form, small, picklable config for LOAD_MODEL (precision,
    # resolution, etc.) — kept generic so new model adapters don't
    # require new Command subclasses.
    config: dict | None = None


@dataclass(frozen=True, slots=True)
class Ack:
    """Sent back from the worker acknowledging a Command."""

    command_type: CommandType
    success: bool
    message: str = ""


@dataclass(frozen=True, slots=True)
class FrameRequest:
    """Sent from the UI process to the worker via the bounded frame queue."""

    session_id: int
    frame_id: int
    image: np.ndarray  # HxWx3 BGR uint8, same convention as core.capture


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Sent back from the worker via the result queue.

    Deliberately flatter than the full ``GeometryResult`` the model
    adapters produce internally — only what needs to cross the process
    boundary and be checked for staleness lives here; the richer
    dataclass stays a worker-process-local concern until Phase 4 wires
    real geometry through.
    """

    session_id: int
    frame_id: int
    model_id: str
    depth: np.ndarray | None
    latency_ms: float
    is_metric: bool = False
    error: str | None = None