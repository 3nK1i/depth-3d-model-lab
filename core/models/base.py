"""Unified model adapter interface (spec §9).

Runs entirely inside the inference worker process — a plain Python ABC
(no QObject involved here, unlike core.capture.base_source, so the
metaclass conflict that forced a workaround there doesn't apply).
Every model family (mock, MoGe-2, Depth Anything V2, ...) implements
this same small surface so the worker's dispatch logic never needs to
know which family is currently loaded.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.models.result import GeometryResult
from core.runtime.messages import FrameRequest


class BaseGeometryModel(ABC):
    """Common lifecycle every model adapter implements."""

    @abstractmethod
    def load(self, device: str, precision: str, config: dict[str, Any]) -> None:
        """Load weights/allocate resources. Called once per model switch."""

    @abstractmethod
    def warmup(self) -> None:
        """Run a throwaway inference to prime CUDA kernels/caches."""

    @abstractmethod
    def infer(self, frame: FrameRequest) -> GeometryResult:
        """Run inference on one frame and return a populated result."""

    @abstractmethod
    def unload(self) -> dict[str, Any] | None:
        """Release resources before switching models or shutting down.

        Implementations may return lightweight cleanup telemetry containing
        only picklable primitives.
        """

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Static info about this model: id, whether it's metric, which
        outputs it populates, etc. Used for UI badges and compatibility
        checks, not for per-frame data."""