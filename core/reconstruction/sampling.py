"""Deterministic frame-sampling policies for offline reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import floor, isfinite

from core.reconstruction.contracts import ReconstructionFrame
from core.reconstruction.errors import ReconstructionValidationError


class FrameSamplingMode(str, Enum):
    EVERY_FRAME = "every_frame"
    FIXED_INTERVAL = "fixed_interval"
    TARGET_FPS = "target_fps"


class FrameSamplingPlanMode(str, Enum):
    """Resolved execution mode for a sampling policy."""

    POLICY_NATIVE = "policy_native"
    UNIFORM_FULL_SPAN = "uniform_full_span"


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


@dataclass(frozen=True, slots=True)
class FrameSamplingPlan:
    """Resolved deterministic plan used by :class:`FrameSampler`.

    TARGET_FPS sampling keeps its historical timestamp-driven behavior while the
    requested rate fits inside ``maximum_sampled_frames``.  If the cap would
    truncate a known finite source, the plan switches to uniformly distributed
    source-frame targets so the beginning, middle, and end of the source are
    represented instead of silently sampling only a prefix.
    """

    mode: FrameSamplingPlanMode
    estimated_total_frames: int | None
    expected_frame_rate_hz: float | None
    estimated_duration_seconds: float | None
    requested_target_fps_hz: float | None
    effective_target_fps_hz: float | None
    maximum_sampled_frames: int | None
    nominal_sample_count: int | None
    planned_sample_count: int | None
    target_sequence_indices: tuple[int, ...] = ()

    @property
    def full_span_cap_active(self) -> bool:
        return self.mode is FrameSamplingPlanMode.UNIFORM_FULL_SPAN

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "estimated_total_frames": self.estimated_total_frames,
            "expected_frame_rate_hz": self.expected_frame_rate_hz,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "requested_target_fps_hz": self.requested_target_fps_hz,
            "effective_target_fps_hz": self.effective_target_fps_hz,
            "maximum_sampled_frames": self.maximum_sampled_frames,
            "nominal_sample_count": self.nominal_sample_count,
            "planned_sample_count": self.planned_sample_count,
            "full_span_cap_active": self.full_span_cap_active,
            "target_sequence_indices": list(self.target_sequence_indices),
        }


def _uniform_indices(total_frames: int, sample_count: int) -> tuple[int, ...]:
    if sample_count <= 0 or total_frames <= 0:
        return ()
    if sample_count == 1:
        return (0,)
    last = total_frames - 1
    indices = tuple(
        int(round(position * last / (sample_count - 1)))
        for position in range(sample_count)
    )
    # When sample_count <= total_frames, rounded linspace indices must be unique.
    # Keep a fail-closed assertion here because duplicate targets would bias the
    # temporal distribution and invalidate provenance.
    if len(indices) != len(set(indices)):
        raise ReconstructionValidationError(
            "Uniform full-span sampling produced duplicate source-frame targets."
        )
    return indices


def build_frame_sampling_plan(
    policy: FrameSamplingPolicy,
    *,
    estimated_total_frames: int | None = None,
    expected_frame_rate_hz: float | None = None,
) -> FrameSamplingPlan:
    """Resolve a sampling policy against finite-source metadata.

    Only capped TARGET_FPS sampling is adapted.  All other policies retain their
    historical one-pass behavior.  The finite-source adaptation is deliberately
    metadata-driven and deterministic; it does not inspect image content or
    reconstruction quality.
    """

    if not isinstance(policy, FrameSamplingPolicy):
        raise TypeError("policy must be a FrameSamplingPolicy.")

    total: int | None = None
    if estimated_total_frames is not None:
        if isinstance(estimated_total_frames, bool) or int(estimated_total_frames) <= 0:
            raise ReconstructionValidationError(
                "estimated_total_frames must be positive when provided."
            )
        total = int(estimated_total_frames)

    source_fps: float | None = None
    if expected_frame_rate_hz is not None:
        source_fps = float(expected_frame_rate_hz)
        if not isfinite(source_fps) or source_fps <= 0.0:
            raise ReconstructionValidationError(
                "expected_frame_rate_hz must be finite and positive when provided."
            )

    requested = (
        float(policy.target_fps_hz)
        if policy.mode is FrameSamplingMode.TARGET_FPS and policy.target_fps_hz is not None
        else None
    )
    maximum = policy.maximum_sampled_frames

    duration: float | None = None
    nominal_count: int | None = None
    if total is not None and source_fps is not None:
        duration = max(0.0, (total - 1) / source_fps)
        if requested is not None:
            nominal_count = min(total, int(floor(duration * requested + 1e-12)) + 1)

    can_uniformly_cap = (
        policy.mode is FrameSamplingMode.TARGET_FPS
        and requested is not None
        and maximum is not None
        and total is not None
        and source_fps is not None
        and nominal_count is not None
        and maximum >= 2
        and nominal_count > maximum
        and total > maximum
    )

    if can_uniformly_cap:
        planned_count = int(maximum)
        indices = _uniform_indices(total, planned_count)
        effective = (
            (planned_count - 1) / duration
            if duration is not None and duration > 0.0 and planned_count > 1
            else None
        )
        return FrameSamplingPlan(
            mode=FrameSamplingPlanMode.UNIFORM_FULL_SPAN,
            estimated_total_frames=total,
            expected_frame_rate_hz=source_fps,
            estimated_duration_seconds=duration,
            requested_target_fps_hz=requested,
            effective_target_fps_hz=effective,
            maximum_sampled_frames=maximum,
            nominal_sample_count=nominal_count,
            planned_sample_count=planned_count,
            target_sequence_indices=indices,
        )

    planned_count: int | None = None
    if nominal_count is not None:
        planned_count = nominal_count if maximum is None else min(nominal_count, maximum)
    return FrameSamplingPlan(
        mode=FrameSamplingPlanMode.POLICY_NATIVE,
        estimated_total_frames=total,
        expected_frame_rate_hz=source_fps,
        estimated_duration_seconds=duration,
        requested_target_fps_hz=requested,
        effective_target_fps_hz=requested,
        maximum_sampled_frames=maximum,
        nominal_sample_count=nominal_count,
        planned_sample_count=planned_count,
        target_sequence_indices=(),
    )


class FrameSampler:
    """Stateful one-pass deterministic sampler with optional full-span planning."""

    def __init__(
        self,
        policy: FrameSamplingPolicy,
        *,
        plan: FrameSamplingPlan | None = None,
    ) -> None:
        if not isinstance(policy, FrameSamplingPolicy):
            raise TypeError("policy must be a FrameSamplingPolicy.")
        if plan is not None and not isinstance(plan, FrameSamplingPlan):
            raise TypeError("plan must be a FrameSamplingPlan or None.")
        self.policy = policy
        self.plan = plan or build_frame_sampling_plan(policy)
        self._selected_count = 0
        self._last_timestamp_ns: int | None = None
        self._uniform_target_cursor = 0

    def _should_select_uniform_full_span(self, frame: ReconstructionFrame) -> bool:
        targets = self.plan.target_sequence_indices
        if self._uniform_target_cursor >= len(targets):
            return False
        target_index = targets[self._uniform_target_cursor]
        if frame.sequence_index < target_index:
            return False
        # If the exact target frame is invalid, choose the first subsequent valid
        # frame instead.  This preserves sample count as far as the source allows
        # without pulling a later target backward or decoding the source twice.
        self._uniform_target_cursor += 1
        while (
            self._uniform_target_cursor < len(targets)
            and targets[self._uniform_target_cursor] <= frame.sequence_index
        ):
            self._uniform_target_cursor += 1
        return True

    def should_select(self, frame: ReconstructionFrame) -> bool:
        if not isinstance(frame, ReconstructionFrame):
            raise TypeError("frame must be a ReconstructionFrame.")
        if not frame.is_valid:
            return False
        maximum = self.policy.maximum_sampled_frames
        if maximum is not None and self._selected_count >= maximum:
            return False

        selected = False
        if self.plan.mode is FrameSamplingPlanMode.UNIFORM_FULL_SPAN:
            selected = self._should_select_uniform_full_span(frame)
        elif self.policy.mode is FrameSamplingMode.EVERY_FRAME:
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
