"""Unified per-frame geometry result shared by every model adapter.

The worker-local result can carry either a depth-only prediction or a direct
point map. Direct point maps may already be regularly sampled to a bounded
point budget; ``point_stride`` records that sampling so the UI can align RGB,
depth and masks without guessing.
"""

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
    latency_ms: float = 0.0
    is_metric: bool = False
    model_id: str = ""
    model_name: str = ""
    precision: str = ""
    vram_used_gib: float | None = None
    vram_total_gib: float | None = None
