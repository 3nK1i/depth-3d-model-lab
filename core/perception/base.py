"""Common lifecycle plus capability-specific perception adapter protocols."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable
from core.perception.messages import FramePerceptionRequest, PromptedSegmentationRequest, SequenceResetRequest
from core.perception.raw import RawAutomaticMaskBatch, RawPerceptionBatch, RawPromptedMask

class PerceptionAdapterError(RuntimeError):
    """Base typed error for perception adapters."""

class UnsupportedPerceptionOperation(PerceptionAdapterError):
    """Typed unsupported-operation error."""

class PerceptionDependencyError(PerceptionAdapterError):
    """Optional dependency is unavailable."""

class PerceptionModelLoadError(PerceptionAdapterError):
    """Model/checkpoint loading failed."""

class BasePerceptionAdapter(ABC):

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

    def infer_frame(self, request: FramePerceptionRequest) -> RawPerceptionBatch:
        ...

@runtime_checkable
class PromptedSegmentationAdapter(Protocol):

    def refine_mask(self, request: PromptedSegmentationRequest) -> RawPromptedMask:
        ...

@runtime_checkable
class AutomaticSegmentationAdapter(Protocol):

    def propose_masks(self, request: FramePerceptionRequest) -> RawAutomaticMaskBatch:
        ...

@runtime_checkable
class SequencePerceptionAdapter(Protocol):

    def reset_sequence(self, request: SequenceResetRequest) -> None:
        ...
