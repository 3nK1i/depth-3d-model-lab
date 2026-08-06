"""Deterministic offline keyframe selection before SfM is introduced."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import cv2
import numpy as np

from core.reconstruction.contracts import (
    KeyframeSelectionReason,
    ReconstructionFrame,
    ReconstructionKeyframe,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.quality import FrameQualityMetrics


@dataclass(frozen=True, slots=True)
class KeyframeSelectionPolicy:
    minimum_spacing_ns: int = 250_000_000
    forced_spacing_ns: int = 1_500_000_000
    minimum_visual_change: float = 0.055
    minimum_quality_score: float = 0.12
    maximum_keyframes: int = 250
    comparison_size: int = 64

    def __post_init__(self) -> None:
        for field_name in ("minimum_spacing_ns", "forced_spacing_ns", "maximum_keyframes", "comparison_size"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ReconstructionValidationError(f"{field_name} must be a positive integer.")
            object.__setattr__(self, field_name, int(value))
        if self.forced_spacing_ns < self.minimum_spacing_ns:
            raise ReconstructionValidationError(
                "forced_spacing_ns must be greater than or equal to minimum_spacing_ns."
            )
        for field_name in ("minimum_visual_change", "minimum_quality_score"):
            value = float(getattr(self, field_name))
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ReconstructionValidationError(f"{field_name} must be between 0 and 1.")
            object.__setattr__(self, field_name, value)


def visual_change_score(first: np.ndarray, second: np.ndarray, *, size: int = 64) -> float:
    if not isinstance(first, np.ndarray) or not isinstance(second, np.ndarray):
        raise TypeError("visual comparison inputs must be NumPy arrays.")

    def _gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ReconstructionValidationError("Keyframe images must be HxW or HxWx3.")
        return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

    a = _gray(first).astype(np.float32)
    b = _gray(second).astype(np.float32)
    return float(np.mean(np.abs(a - b)) / 255.0)


class KeyframeSelector:
    def __init__(self, policy: KeyframeSelectionPolicy) -> None:
        if not isinstance(policy, KeyframeSelectionPolicy):
            raise TypeError("policy must be a KeyframeSelectionPolicy.")
        self.policy = policy
        self._last_frame: ReconstructionFrame | None = None
        self._last_image: np.ndarray | None = None
        self._keyframes: list[ReconstructionKeyframe] = []

    def consider(
        self,
        frame: ReconstructionFrame,
        image: np.ndarray,
        quality: FrameQualityMetrics,
    ) -> ReconstructionKeyframe | None:
        if not isinstance(frame, ReconstructionFrame):
            raise TypeError("frame must be a ReconstructionFrame.")
        if not frame.is_valid or not quality.accepted:
            return None
        if quality.quality_score < self.policy.minimum_quality_score:
            return None
        if len(self._keyframes) >= self.policy.maximum_keyframes:
            return None

        if self._last_frame is None:
            return self._select(frame, image, KeyframeSelectionReason.FIRST_FRAME, quality.quality_score)

        elapsed = frame.timestamp_ns - self._last_frame.timestamp_ns
        if elapsed < self.policy.minimum_spacing_ns:
            return None
        change = visual_change_score(self._last_image, image, size=self.policy.comparison_size)  # type: ignore[arg-type]
        if change >= self.policy.minimum_visual_change:
            reason = KeyframeSelectionReason.VISUAL_CHANGE
        elif elapsed >= self.policy.forced_spacing_ns:
            reason = KeyframeSelectionReason.TEMPORAL_SPACING
        else:
            return None
        score = float(np.clip(0.65 * change + 0.35 * quality.quality_score, 0.0, 1.0))
        return self._select(frame, image, reason, score)

    def _select(
        self,
        frame: ReconstructionFrame,
        image: np.ndarray,
        reason: KeyframeSelectionReason,
        score: float,
    ) -> ReconstructionKeyframe:
        sequence = len(self._keyframes)
        keyframe = ReconstructionKeyframe(
            keyframe_id=f"{frame.session_id}:keyframe:{sequence:08d}",
            session_id=frame.session_id,
            frame_id=frame.frame_id,
            selection_sequence=sequence,
            reason=reason,
            score=float(np.clip(score, 0.0, 1.0)),
        )
        self._keyframes.append(keyframe)
        self._last_frame = frame
        self._last_image = np.ascontiguousarray(image).copy()
        self._last_image.setflags(write=False)
        return keyframe

    @property
    def keyframes(self) -> tuple[ReconstructionKeyframe, ...]:
        return tuple(self._keyframes)
