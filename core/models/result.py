"""Unified per-frame geometry result shared by every model adapter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class GeometryResult:
    session_id: int
    frame_id: int
    timestamp: float
    rgb: np.ndarray
    depth: np.ndarray | None
    points: np.ndarray | None = None
    colors: np.ndarray | None = None
    normals: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    confidence: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    point_stride: int = 1

    # Model forward-pass latency (CUDA events on GPU, wall clock on CPU).
    latency_ms: float = 0.0
    preprocess_ms: float | None = None
    postprocess_ms: float | None = None
    worker_total_ms: float | None = None
    queue_wait_ms: float | None = None

    is_metric: bool = False
    model_id: str = ""
    model_name: str = ""
    precision: str = ""

    # CUDA memory values are explicit so the UI never mislabels reserved
    # allocator memory as allocated tensor memory.
    vram_used_gib: float | None = None
    vram_allocated_gib: float | None = None
    vram_reserved_gib: float | None = None
    vram_peak_reserved_gib: float | None = None
    vram_device_used_gib: float | None = None
    vram_total_gib: float | None = None
