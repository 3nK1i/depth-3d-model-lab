"""Phase 8.5D hardening for optimized live trajectories.

The coordinator in this module adds worker-generation/optimizer-epoch handling,
raw-pose integrity checks, bounded quality assessment, and stable telemetry on
top of :mod:`core.reconstruction.live_trajectory`.

A restarted native SLAM worker may legitimately reset its keyframe IDs, map
revision, and optimized-pose revision.  The coordinator converts that process
restart into a new optimizer epoch while preserving every raw tracking pose.
This prevents corrections from an older worker instance from leaking into the
new optimizer state.

No metric/world alignment, dense fusion, or UI behavior is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from threading import RLock

import numpy as np

from core.contracts.validation import require_finite
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.live_contracts import LiveSlamObservation
from core.reconstruction.live_global_consistency import NativeGlobalConsistencyStatus
from core.reconstruction.live_trajectory import (
    LiveOptimizedTrajectory,
    LiveTrajectoryEpochTransition,
    LiveTrajectoryPoseLayer,
    LiveTrajectoryRevisionUpdate,
    LiveTrajectorySnapshot,
)
from core.reconstruction.validation import require_nonnegative_int, require_positive_int, require_uuid


@dataclass(frozen=True, slots=True)
class LiveTrajectoryHardeningPolicy:
    """Acceptance policy for one live global-consistency trajectory."""

    min_keyframe_match_ratio: float = 0.70
    min_exact_optimized_keyframes: int = 4
    require_raw_tracking_integrity: bool = True

    def __post_init__(self) -> None:
        ratio = require_finite(
            self.min_keyframe_match_ratio,
            field_name="min_keyframe_match_ratio",
        )
        if not 0.0 <= ratio <= 1.0:
            raise ReconstructionValidationError(
                "min_keyframe_match_ratio must be between 0 and 1."
            )
        object.__setattr__(self, "min_keyframe_match_ratio", ratio)
        object.__setattr__(
            self,
            "min_exact_optimized_keyframes",
            require_nonnegative_int(
                self.min_exact_optimized_keyframes,
                field_name="min_exact_optimized_keyframes",
            ),
        )
        object.__setattr__(
            self,
            "require_raw_tracking_integrity",
            bool(self.require_raw_tracking_integrity),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "min_keyframe_match_ratio": self.min_keyframe_match_ratio,
            "min_exact_optimized_keyframes": self.min_exact_optimized_keyframes,
            "require_raw_tracking_integrity": self.require_raw_tracking_integrity,
        }


@dataclass(frozen=True, slots=True)
class LiveTrajectoryQualityAssessment:
    session_id: str
    optimizer_epoch: int
    worker_generation: int
    raw_pose_count: int
    native_keyframe_count: int
    exact_optimized_keyframes: int
    unmatched_keyframes: int
    keyframe_match_ratio: float
    optimization_revision: int
    raw_tracking_integrity: bool
    accepted: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", require_uuid(self.session_id, field_name="session_id"))
        object.__setattr__(self, "optimizer_epoch", require_positive_int(self.optimizer_epoch, field_name="optimizer_epoch"))
        object.__setattr__(self, "worker_generation", require_nonnegative_int(self.worker_generation, field_name="worker_generation"))
        for name in (
            "raw_pose_count",
            "native_keyframe_count",
            "exact_optimized_keyframes",
            "unmatched_keyframes",
            "optimization_revision",
        ):
            object.__setattr__(
                self,
                name,
                require_nonnegative_int(getattr(self, name), field_name=name),
            )
        ratio = require_finite(self.keyframe_match_ratio, field_name="keyframe_match_ratio")
        if not 0.0 <= ratio <= 1.0:
            raise ReconstructionValidationError("keyframe_match_ratio must be between 0 and 1.")
        object.__setattr__(self, "keyframe_match_ratio", ratio)
        reasons = tuple(str(item).strip() for item in self.reasons)
        if any(not item for item in reasons):
            raise ReconstructionValidationError("Quality assessment reasons must be non-empty.")
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "raw_tracking_integrity", bool(self.raw_tracking_integrity))
        object.__setattr__(self, "accepted", bool(self.accepted))

    def as_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "optimizer_epoch": self.optimizer_epoch,
            "worker_generation": self.worker_generation,
            "raw_pose_count": self.raw_pose_count,
            "native_keyframe_count": self.native_keyframe_count,
            "exact_optimized_keyframes": self.exact_optimized_keyframes,
            "unmatched_keyframes": self.unmatched_keyframes,
            "keyframe_match_ratio": self.keyframe_match_ratio,
            "optimization_revision": self.optimization_revision,
            "raw_tracking_integrity": self.raw_tracking_integrity,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class LiveTrajectoryHardeningTelemetry:
    worker_generation: int
    optimizer_epoch: int
    observations_ingested: int
    statuses_applied: int
    status_rejections: int
    epoch_transitions: int
    generation_regressions_rejected: int
    raw_integrity_failures: int
    raw_pose_count: int
    optimized_keyframe_count: int
    map_revision: int
    optimization_revision: int

    def as_dict(self) -> dict[str, int]:
        return {
            "worker_generation": self.worker_generation,
            "optimizer_epoch": self.optimizer_epoch,
            "observations_ingested": self.observations_ingested,
            "statuses_applied": self.statuses_applied,
            "status_rejections": self.status_rejections,
            "epoch_transitions": self.epoch_transitions,
            "generation_regressions_rejected": self.generation_regressions_rejected,
            "raw_integrity_failures": self.raw_integrity_failures,
            "raw_pose_count": self.raw_pose_count,
            "optimized_keyframe_count": self.optimized_keyframe_count,
            "map_revision": self.map_revision,
            "optimization_revision": self.optimization_revision,
        }


class LiveTrajectoryCoordinator:
    """Worker-generation-aware facade around :class:`LiveOptimizedTrajectory`."""

    def __init__(
        self,
        *,
        session_id: str,
        policy: LiveTrajectoryHardeningPolicy | None = None,
        timestamp_match_tolerance_ns: int = 2_000,
    ) -> None:
        self._session_id = require_uuid(session_id, field_name="session_id")
        self._policy = policy or LiveTrajectoryHardeningPolicy()
        if not isinstance(self._policy, LiveTrajectoryHardeningPolicy):
            raise TypeError("policy must be LiveTrajectoryHardeningPolicy or None.")
        self._trajectory = LiveOptimizedTrajectory(
            session_id=self._session_id,
            timestamp_match_tolerance_ns=timestamp_match_tolerance_ns,
        )
        self._lock = RLock()
        self._worker_generation = 0
        self._observations_ingested = 0
        self._statuses_applied = 0
        self._status_rejections = 0
        self._epoch_transitions = 0
        self._generation_regressions_rejected = 0
        self._raw_integrity_failures = 0
        self._raw_pose_digests: dict[str, tuple[str, str]] = {}

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def trajectory(self) -> LiveOptimizedTrajectory:
        return self._trajectory

    @property
    def worker_generation(self) -> int:
        with self._lock:
            return self._worker_generation

    @property
    def policy(self) -> LiveTrajectoryHardeningPolicy:
        return self._policy

    def _ensure_worker_generation(self, worker_generation: int) -> LiveTrajectoryEpochTransition | None:
        generation = require_nonnegative_int(worker_generation, field_name="worker_generation")
        with self._lock:
            if generation < self._worker_generation:
                self._generation_regressions_rejected += 1
                raise ReconstructionValidationError(
                    "Stale live-SLAM worker generation was rejected."
                )
            if generation == self._worker_generation:
                return None
            previous_generation = self._worker_generation
            transition = self._trajectory.begin_optimizer_epoch(
                reason=f"worker_generation:{previous_generation}->{generation}"
            )
            self._worker_generation = generation
            self._epoch_transitions += 1
            return transition

    def begin_worker_generation(
        self,
        worker_generation: int,
    ) -> LiveTrajectoryEpochTransition | None:
        """Advance worker generation immediately when a replacement worker starts.

        Callers should invoke this lifecycle hook as soon as a new native worker
        generation is established, before the replacement worker is expected to
        emit a tracking pose.  Startup observations may legitimately be
        ``INITIALIZING`` with ``pose=None``; tying epoch invalidation to the first
        pose would leave previous-generation optimizer corrections visible for a
        short but unsafe window.
        """

        return self._ensure_worker_generation(worker_generation)

    def ingest_observation(
        self,
        observation: LiveSlamObservation,
        *,
        worker_generation: int = 0,
    ) -> bool:
        self._ensure_worker_generation(worker_generation)
        inserted = self._trajectory.ingest_observation(observation)
        if inserted and observation.pose is not None:
            matrix_bytes = np.asarray(
                observation.pose.T_reconstruction_from_camera,
                dtype=np.float64,
            ).tobytes()
            digest = sha256(matrix_bytes).hexdigest()
            with self._lock:
                self._raw_pose_digests[observation.frame_id] = (
                    observation.pose.pose_id,
                    digest,
                )
                self._observations_ingested += 1
        return inserted

    def apply_native_status(
        self,
        status: NativeGlobalConsistencyStatus,
        *,
        worker_generation: int = 0,
    ) -> LiveTrajectoryRevisionUpdate:
        self._ensure_worker_generation(worker_generation)
        try:
            update = self._trajectory.apply_native_status(status)
        except ReconstructionValidationError:
            with self._lock:
                self._status_rejections += 1
            raise
        with self._lock:
            self._statuses_applied += 1
        return update

    def raw_tracking_integrity_ok(self) -> bool:
        with self._lock:
            expected = dict(self._raw_pose_digests)
        for frame_id, (pose_id, expected_digest) in expected.items():
            raw = self._trajectory.raw_pose_for_frame(frame_id)
            if raw is None or raw.raw_pose_id != pose_id:
                with self._lock:
                    self._raw_integrity_failures += 1
                return False
            digest = sha256(
                np.asarray(raw.T_reconstruction_from_camera, dtype=np.float64).tobytes()
            ).hexdigest()
            if digest != expected_digest or raw.layer is not LiveTrajectoryPoseLayer.TRACKING:
                with self._lock:
                    self._raw_integrity_failures += 1
                return False
        return True

    def assess_quality(
        self,
        status: NativeGlobalConsistencyStatus,
    ) -> LiveTrajectoryQualityAssessment:
        if not isinstance(status, NativeGlobalConsistencyStatus):
            raise TypeError("status must be NativeGlobalConsistencyStatus.")
        if status.session_id != self._session_id:
            raise ReconstructionValidationError("Quality status session_id mismatch.")
        snapshot = self._trajectory.snapshot()
        native_keyframes = status.keyframe_count
        exact = snapshot.optimized_keyframe_count
        unmatched = max(0, native_keyframes - exact)
        ratio = 0.0 if native_keyframes <= 0 else min(1.0, exact / native_keyframes)
        raw_integrity = self.raw_tracking_integrity_ok()
        reasons: list[str] = []
        if snapshot.optimization_revision <= 0:
            reasons.append("no_optimized_revision")
        if exact < self._policy.min_exact_optimized_keyframes:
            reasons.append("insufficient_exact_optimized_keyframes")
        if ratio < self._policy.min_keyframe_match_ratio:
            reasons.append("insufficient_keyframe_match_ratio")
        if self._policy.require_raw_tracking_integrity and not raw_integrity:
            reasons.append("raw_tracking_integrity_failed")
        accepted = not reasons
        return LiveTrajectoryQualityAssessment(
            session_id=self._session_id,
            optimizer_epoch=snapshot.optimizer_epoch,
            worker_generation=self.worker_generation,
            raw_pose_count=snapshot.raw_pose_count,
            native_keyframe_count=native_keyframes,
            exact_optimized_keyframes=exact,
            unmatched_keyframes=unmatched,
            keyframe_match_ratio=ratio,
            optimization_revision=snapshot.optimization_revision,
            raw_tracking_integrity=raw_integrity,
            accepted=accepted,
            reasons=tuple(reasons),
        )

    def snapshot(self, *, allow_propagated: bool = True) -> LiveTrajectorySnapshot:
        return self._trajectory.snapshot(allow_propagated=allow_propagated)

    @property
    def telemetry(self) -> LiveTrajectoryHardeningTelemetry:
        snapshot = self._trajectory.snapshot()
        with self._lock:
            return LiveTrajectoryHardeningTelemetry(
                worker_generation=self._worker_generation,
                optimizer_epoch=snapshot.optimizer_epoch,
                observations_ingested=self._observations_ingested,
                statuses_applied=self._statuses_applied,
                status_rejections=self._status_rejections,
                epoch_transitions=self._epoch_transitions,
                generation_regressions_rejected=self._generation_regressions_rejected,
                raw_integrity_failures=self._raw_integrity_failures,
                raw_pose_count=snapshot.raw_pose_count,
                optimized_keyframe_count=snapshot.optimized_keyframe_count,
                map_revision=snapshot.map_revision,
                optimization_revision=snapshot.optimization_revision,
            )
