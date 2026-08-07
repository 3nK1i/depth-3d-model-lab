"""Hardening contracts for Phase 8.4D live visual SLAM.

These policies stay dependency-light and deliberately separate worker/process
recovery from capture ownership.  They do not enable loop closure, global
optimization, dense fusion, or the Reconstruction UI.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.contracts.validation import require_finite
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.validation import require_nonnegative_int


@dataclass(frozen=True, slots=True)
class LiveSlamRecoveryPolicy:
    """Bounded recovery policy for one live-SLAM runtime session."""

    max_worker_restarts: int = 2
    restart_backoff_s: float = 0.10
    max_consecutive_stale_frames: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_worker_restarts",
            require_nonnegative_int(
                self.max_worker_restarts,
                field_name="max_worker_restarts",
            ),
        )
        backoff = require_finite(self.restart_backoff_s, field_name="restart_backoff_s")
        if backoff < 0.0:
            raise ReconstructionValidationError("restart_backoff_s must be non-negative.")
        object.__setattr__(self, "restart_backoff_s", backoff)
        object.__setattr__(
            self,
            "max_consecutive_stale_frames",
            require_nonnegative_int(
                self.max_consecutive_stale_frames,
                field_name="max_consecutive_stale_frames",
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "max_worker_restarts": self.max_worker_restarts,
            "restart_backoff_s": self.restart_backoff_s,
            "max_consecutive_stale_frames": self.max_consecutive_stale_frames,
        }


@dataclass(frozen=True, slots=True)
class LiveSlamHardeningTelemetry:
    """Small stable telemetry snapshot suitable for future UI/exports."""

    offered_frames: int
    processed_frames: int
    replaced_frames: int
    rejected_frames: int
    observations: int
    stale_frames: int
    worker_restarts: int
    worker_restart_failures: int
    tracking_loss_events: int
    tracking_recoveries: int
    stop_escalations: int
    last_processing_latency_ns: int | None
    max_processing_latency_ns: int
    mean_processing_latency_ns: float | None
    cleanup_complete: bool

    @property
    def replacement_ratio(self) -> float:
        if self.offered_frames <= 0:
            return 0.0
        return self.replaced_frames / self.offered_frames

    def as_dict(self) -> dict[str, object]:
        return {
            "offered_frames": self.offered_frames,
            "processed_frames": self.processed_frames,
            "replaced_frames": self.replaced_frames,
            "rejected_frames": self.rejected_frames,
            "observations": self.observations,
            "stale_frames": self.stale_frames,
            "worker_restarts": self.worker_restarts,
            "worker_restart_failures": self.worker_restart_failures,
            "tracking_loss_events": self.tracking_loss_events,
            "tracking_recoveries": self.tracking_recoveries,
            "stop_escalations": self.stop_escalations,
            "last_processing_latency_ns": self.last_processing_latency_ns,
            "max_processing_latency_ns": self.max_processing_latency_ns,
            "mean_processing_latency_ns": self.mean_processing_latency_ns,
            "replacement_ratio": self.replacement_ratio,
            "cleanup_complete": self.cleanup_complete,
        }
