"""Common adapter interface for every depth and geometry model.

Adapters run entirely inside the isolated inference worker. Model discovery,
construction, capabilities, dependencies, and presentation metadata belong to
the model registry; controllers and the worker never branch on model families.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.models.result import GeometryResult
from core.runtime.messages import FrameRequest


class BaseGeometryModel(ABC):
    """Small lifecycle contract implemented by every registered adapter."""

    @abstractmethod
    def load(self, device: str, precision: str, config: dict[str, Any]) -> None:
        """Load weights and allocate resources for one registry-resolved config."""

    @abstractmethod
    def warmup(self) -> None:
        """Run throwaway inference to prime kernels and caches."""

    @abstractmethod
    def infer(self, frame: FrameRequest) -> GeometryResult:
        """Run one frame and return the normalized geometry result contract."""

    def reset_sequence(self) -> dict[str, Any] | None:
        """Reset optional temporal state while keeping model weights loaded.

        Stateless Phase 5/6 adapters inherit this no-op implementation. Stateful
        adapters override it. The generic worker calls it on STOP, so temporal
        caches never leak into a restarted source session.
        """

        return None

    @abstractmethod
    def unload(self) -> dict[str, Any] | None:
        """Release resources and optionally return primitive cleanup telemetry."""

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return registry metadata enriched with runtime-only facts.

        Static capabilities and presentation fields must originate from the
        registry. Adapters may add values only known after load, such as the
        selected checkpoint, device, precision, and warm-up configuration.
        """
