"""Common lifecycle plus capability-specific perception adapter protocols."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from core.perception.messages import (
    FramePerceptionRequest,
    PromptedSegmentationRequest,
    SequenceResetRequest,
)


class PerceptionAdapterError(RuntimeError):
    """Base typed error for perception adapters."""


class UnsupportedPerceptionOperation(PerceptionAdapterError):
    """Raised when a typed request is unsupported by an adapter capability."""


class BasePerceptionAdapter(ABC):
    """Lifecycle-only base; task operations live in capability protocols."""

    @abstractmethod
    def load(self, config: dict[str, object]) -> None:
        raise NotImplementedError

    @abstractmethod
    def warmup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        raise NotImplementedError


@runtime_checkable
class FramePerceptionAdapter(Protocol):
    def infer_frame(self, request: FramePerceptionRequest) -> Any:
        """Return a PerceptionFrameResult for one immutable frame request."""


@runtime_checkable
class PromptedSegmentationAdapter(Protocol):
    def refine_mask(self, request: PromptedSegmentationRequest) -> Any:
        """Return a PromptedMaskRefinement for a typed prompt request."""


@runtime_checkable
class SequencePerceptionAdapter(Protocol):
    def reset_sequence(self, request: SequenceResetRequest) -> None:
        """Reset model-native sequence state without model-ID branching."""
