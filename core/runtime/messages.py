"""Picklable IPC message types between the UI and inference processes.

Only primitives and NumPy arrays cross this boundary. No Qt objects, live CUDA
handles or model instances are allowed here.
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
    config: dict | None = None


@dataclass(frozen=True, slots=True)
class Ack:
    """Worker acknowledgement for a control command.

    ``metadata`` is populated after a successful model load. Defaults keep the
    command contract backward-compatible with existing tests and callers.
    """

    command_type: CommandType
    success: bool
    message: str = ""
    model_id: str = ""
    metadata: dict | None = None
    load_time_ms: float | None = None


@dataclass(frozen=True, slots=True)
class FrameRequest:
    """One latest-frame inference request.

    ``point_budget`` lets direct-geometry adapters downsample point maps in the
    worker before IPC. This avoids copying a full H×W×3 float map for every live
    frame while preserving a full-resolution depth map for the depth panel.
    """

    session_id: int
    frame_id: int
    image: np.ndarray  # HxWx3 BGR uint8
    point_budget: int | None = None


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Geometry data returned by the worker.

    ``points``/``normals`` may be sampled regular grids. ``point_stride`` is the
    source-image stride used to create them. ``valid_mask`` remains full-size so
    both the depth panel and sampled 3D path can use the same prediction mask.
    """

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
    vram_used_gib: float | None = None
    vram_total_gib: float | None = None
    error: str | None = None
