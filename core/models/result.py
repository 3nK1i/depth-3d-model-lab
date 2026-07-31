"""Unified per-frame geometry result, shared by every model adapter.

Lives worker-process-side. For models that return points/normals/
intrinsics directly, those fields are populated; for depth-only models,
they stay None and Phase 4's projection code derives points from depth
+ a virtual FOV (spec §9).
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
    latency_ms: float = 0.0
    is_metric: bool = False
    model_id: str = ""