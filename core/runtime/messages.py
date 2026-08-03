"""Picklable IPC message types between the UI and inference processes.

Only primitives and NumPy arrays cross this boundary. No Qt objects, live CUDA
handles or model instances are allowed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from core.runtime.failures import FailureInfo


class CommandType(Enum):
    LOAD_MODEL = "load_model"
    UNLOAD_MODEL = "unload_model"
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    SHUTDOWN = "shutdown"


class ModelStage(Enum):
    """Fine-grained worker progress for model lifecycle UI feedback."""

    LOADING_WEIGHTS = "loading_weights"
    WARMING_UP = "warming_up"
    UNLOADING = "unloading"


@dataclass(frozen=True, slots=True)
class ModelStageEvent:
    """Progress event emitted before a potentially long worker operation."""

    stage: ModelStage
    model_id: str = ""
    message: str = ""


@dataclass(frozen=True, slots=True)
class Command:
    """Sent from the UI process to the worker over the control pipe."""

    type: CommandType
    session_id: int = 0
    model_id: str = ""
    config: dict | None = None


@dataclass(frozen=True, slots=True)
class Ack:
    """Worker acknowledgement for a control command.

    ``load_time_ms`` remains the total load + warm-up duration for backward
    compatibility. The explicit timing fields provide the accurate load, warm-up and unload breakdown.
    """

    command_type: CommandType
    success: bool
    message: str = ""
    model_id: str = ""
    metadata: dict | None = None
    load_time_ms: float | None = None
    model_load_time_ms: float | None = None
    warmup_time_ms: float | None = None
    unload_time_ms: float | None = None
    failure: FailureInfo | None = None


@dataclass(frozen=True, slots=True)
class FrameRequest:
    """One latest-frame inference request.

    ``point_budget`` lets direct-geometry adapters downsample point maps in the
    worker before IPC. ``submitted_at`` and ``source_timestamp`` use the host's
    monotonic clock and are telemetry-only; inference correctness never depends
    on them.
    """

    session_id: int
    frame_id: int
    image: np.ndarray  # HxWx3 BGR uint8
    point_budget: int | None = None
    submitted_at: float | None = None
    source_timestamp: float | None = None


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Geometry and measured worker telemetry returned by the worker."""

    session_id: int
    frame_id: int
    model_id: str
    depth: np.ndarray | None
    latency_ms: float
    points: np.ndarray | None = None
    normals: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    point_stride: int = 1
    is_metric: bool = False
    model_name: str = ""
    precision: str = ""

    # Worker timing breakdown. ``latency_ms`` is the model forward pass and is
    # retained as the stable public field used by older tests/callers.
    preprocess_ms: float | None = None
    postprocess_ms: float | None = None
    worker_total_ms: float | None = None
    queue_wait_ms: float | None = None

    # Explicit CUDA allocator/device telemetry. ``vram_used_gib`` remains a
    # compatibility alias and is populated with reserved memory by the worker.
    vram_used_gib: float | None = None
    vram_allocated_gib: float | None = None
    vram_reserved_gib: float | None = None
    vram_peak_reserved_gib: float | None = None
    vram_device_used_gib: float | None = None
    vram_total_gib: float | None = None

    error: str | None = None
    failure: FailureInfo | None = None
