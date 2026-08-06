"""Bounded decoded-frame cache for reconstruction ingestion."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.profiles import (
    ReconstructionMemoryPolicy,
    ReconstructionQualityProfile,
)

_MIB = 1024**2
_GIB = 1024**3


@dataclass(frozen=True, slots=True)
class IngestionCachePolicy:
    maximum_cpu_bytes: int
    maximum_frames: int
    gpu_staging_bytes: int

    def __post_init__(self) -> None:
        for field_name in ("maximum_cpu_bytes", "maximum_frames", "gpu_staging_bytes"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ReconstructionValidationError(
                    f"{field_name} must be a non-negative integer."
                )
            object.__setattr__(self, field_name, int(value))
        if self.maximum_cpu_bytes <= 0 or self.maximum_frames <= 0:
            raise ReconstructionValidationError(
                "CPU cache bytes and frame count must be positive."
            )

    def as_dict(self) -> dict[str, int]:
        return {
            "maximum_cpu_bytes": self.maximum_cpu_bytes,
            "maximum_frames": self.maximum_frames,
            "gpu_staging_bytes": self.gpu_staging_bytes,
        }


def resolve_ingestion_cache_policy(
    memory_policy: ReconstructionMemoryPolicy,
) -> IngestionCachePolicy:
    """Derive a conservative decoded-frame cache from Phase 8.1 budgets."""

    if not isinstance(memory_policy, ReconstructionMemoryPolicy):
        raise TypeError("memory_policy must be a ReconstructionMemoryPolicy.")
    caps = {
        ReconstructionQualityProfile.EFFICIENT: (512 * _MIB, 48),
        ReconstructionQualityProfile.BALANCED: (1536 * _MIB, 160),
        ReconstructionQualityProfile.HIGH_QUALITY: (4 * _GIB, 512),
    }
    byte_cap, frame_cap = caps[memory_policy.profile]
    cpu_fraction = max(128 * _MIB, memory_policy.cpu_budget_bytes // 8)
    return IngestionCachePolicy(
        maximum_cpu_bytes=min(byte_cap, cpu_fraction),
        maximum_frames=min(frame_cap, memory_policy.max_active_keyframes),
        gpu_staging_bytes=memory_policy.frame_cache_bytes,
    )


@dataclass(frozen=True, slots=True)
class FrameCacheStats:
    current_bytes: int
    peak_bytes: int
    current_frames: int
    peak_frames: int
    evictions: int
    rejected_oversized_frames: int

    def as_dict(self) -> dict[str, int]:
        return {
            "current_bytes": self.current_bytes,
            "peak_bytes": self.peak_bytes,
            "current_frames": self.current_frames,
            "peak_frames": self.peak_frames,
            "evictions": self.evictions,
            "rejected_oversized_frames": self.rejected_oversized_frames,
        }


class BoundedFrameCache:
    """Insertion-ordered LRU-like cache with strict byte and item bounds."""

    def __init__(self, policy: IngestionCachePolicy) -> None:
        if not isinstance(policy, IngestionCachePolicy):
            raise TypeError("policy must be an IngestionCachePolicy.")
        self.policy = policy
        self._items: OrderedDict[str, np.ndarray] = OrderedDict()
        self._current_bytes = 0
        self._peak_bytes = 0
        self._peak_frames = 0
        self._evictions = 0
        self._rejected_oversized = 0

    def put(self, frame_id: str, image: np.ndarray) -> tuple[str, ...]:
        key = str(frame_id).strip()
        if not key:
            raise ReconstructionValidationError("frame_id must not be empty.")
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy array.")
        candidate = np.ascontiguousarray(image).copy()
        size = int(candidate.nbytes)
        if size <= 0:
            raise ReconstructionValidationError("Cached images must contain pixels.")
        if size > self.policy.maximum_cpu_bytes:
            self._rejected_oversized += 1
            return ()

        if key in self._items:
            previous = self._items.pop(key)
            self._current_bytes -= int(previous.nbytes)

        evicted: list[str] = []
        while self._items and (
            self._current_bytes + size > self.policy.maximum_cpu_bytes
            or len(self._items) + 1 > self.policy.maximum_frames
        ):
            old_id, old_image = self._items.popitem(last=False)
            self._current_bytes -= int(old_image.nbytes)
            self._evictions += 1
            evicted.append(old_id)

        candidate.setflags(write=False)
        self._items[key] = candidate
        self._current_bytes += size
        self._peak_bytes = max(self._peak_bytes, self._current_bytes)
        self._peak_frames = max(self._peak_frames, len(self._items))
        return tuple(evicted)

    def get(self, frame_id: str) -> np.ndarray | None:
        key = str(frame_id).strip()
        image = self._items.get(key)
        if image is None:
            return None
        self._items.move_to_end(key)
        return image

    def items(self) -> tuple[tuple[str, np.ndarray], ...]:
        return tuple(self._items.items())

    @property
    def stats(self) -> FrameCacheStats:
        return FrameCacheStats(
            current_bytes=self._current_bytes,
            peak_bytes=self._peak_bytes,
            current_frames=len(self._items),
            peak_frames=self._peak_frames,
            evictions=self._evictions,
            rejected_oversized_frames=self._rejected_oversized,
        )
