"""Deterministic frame-sampling policies for offline reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from core.reconstruction.contracts import ReconstructionFrame
from core.reconstruction.errors import ReconstructionValidationError


class FrameSamplingMode(str, Enum):
    EVERY_FRAME = "every_frame"
    FIXED_INTERVAL = "fixed_interval"
    TARGET_FPS = "target_fps"


@dataclass(frozen=True, slots=True)
class FrameSamplingPolicy:
    mode: FrameSamplingMode = FrameSamplingMode.EVERY_FRAME
    frame_interval: int = 1
    target_fps_hz: float | None = None
    maximum_sampled_frames: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, FrameSamplingMode):
            raise TypeError("mode must be a FrameSamplingMode.")
        if isinstance(self.frame_interval, bool) or int(self.frame_interval) <= 0:
            raise ReconstructionValidationError("frame_interval must be a positive integer.")
        object.__setattr__(self, "frame_interval", int(self.frame_interval))
        if self.target_fps_hz is not None:
            rate = float(self.target_fps_hz)
            if not isfinite(rate) or rate <= 0.0:
                raise ReconstructionValidationError("target_fps_hz must be finite and positive.")
            object.__setattr__(self, "target_fps_hz", rate)
        if self.mode is FrameSamplingMode.TARGET_FPS and self.target_fps_hz is None:
            raise ReconstructionValidationError("TARGET_FPS sampling requires target_fps_hz.")
        if self.mode is not FrameSamplingMode.TARGET_FPS and self.target_fps_hz is not None:
            raise ReconstructionValidationError(
                "target_fps_hz is valid only for TARGET_FPS sampling."
            )
        if self.maximum_sampled_frames is not None:
            value = self.maximum_sampled_frames
            if isinstance(value, bool) or int(value) <= 0:
                raise ReconstructionValidationError(
                    "maximum_sampled_frames must be a positive integer when provided."
                )
            object.__setattr__(self, "maximum_sampled_frames", int(value))


class FrameSampler:
    """Stateful one-pass deterministic sampler."""

    def __init__(self, policy: FrameSamplingPolicy) -> None:
        if not isinstance(policy, FrameSamplingPolicy):
            raise TypeError("policy must be a FrameSamplingPolicy.")
        self.policy = policy
        self._selected_count = 0
        self._last_timestamp_ns: int | None = None

    def should_select(self, frame: ReconstructionFrame) -> bool:
        if not isinstance(frame, ReconstructionFrame):
            raise TypeError("frame must be a ReconstructionFrame.")
        if not frame.is_valid:
            return False
        maximum = self.policy.maximum_sampled_frames
        if maximum is not None and self._selected_count >= maximum:
            return False

        selected = False
        if self.policy.mode is FrameSamplingMode.EVERY_FRAME:
            selected = True
        elif self.policy.mode is FrameSamplingMode.FIXED_INTERVAL:
            selected = frame.sequence_index % self.policy.frame_interval == 0
        elif self._last_timestamp_ns is None:
            selected = True
        else:
            interval_ns = int(round(1_000_000_000 / float(self.policy.target_fps_hz)))
            selected = frame.timestamp_ns - self._last_timestamp_ns >= interval_ns

        if selected:
            self._selected_count += 1
            self._last_timestamp_ns = frame.timestamp_ns
        return selected

    @property
    def selected_count(self) -> int:
        return self._selected_count
