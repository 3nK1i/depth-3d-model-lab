"""Native live-global-consistency contracts for Phase 8.5B.

These contracts report what the isolated stella_vslam worker actually owns:
loop-detector enablement, global-optimization activity, current optimized
keyframe poses, and native tracking relocalization observations.  They do not
replace raw tracking poses and never introduce metric scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from core.reconstruction.errors import LiveSlamProtocolError, ReconstructionValidationError
from core.reconstruction.global_consistency import OptimizedKeyframePose
from core.reconstruction.sfm_contracts import ReconstructionGauge, SfmScaleStatus, SfmWorldAlignmentStatus
from core.reconstruction.validation import require_nonnegative_int, require_positive_int, require_uuid


@dataclass(frozen=True, slots=True)
class NativeLoopClosureRequestResult:
    session_id: str
    from_keyframe_id: str
    to_keyframe_id: str
    accepted: bool
    accepted_request_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", require_uuid(self.session_id, field_name="session_id"))
        source = str(self.from_keyframe_id).strip()
        target = str(self.to_keyframe_id).strip()
        if not source or not target or source == target:
            raise ReconstructionValidationError(
                "Native loop-closure request requires two different keyframe IDs."
            )
        object.__setattr__(self, "from_keyframe_id", source)
        object.__setattr__(self, "to_keyframe_id", target)
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be bool.")
        object.__setattr__(
            self,
            "accepted_request_count",
            require_nonnegative_int(self.accepted_request_count, field_name="accepted_request_count"),
        )


@dataclass(frozen=True, slots=True)
class NativeGlobalConsistencyStatus:
    session_id: str
    loop_detector_enabled: bool
    global_optimization_enabled: bool
    loop_ba_running: bool
    map_revision: int
    optimized_pose_revision: int
    keyframe_count: int
    landmark_count: int
    accepted_loop_closure_requests: int
    relocalization_count: int
    last_relocalization_frame_id: str
    optimized_keyframe_poses: tuple[OptimizedKeyframePose, ...]
    gauge: ReconstructionGauge = field(default_factory=ReconstructionGauge)

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", require_uuid(self.session_id, field_name="session_id"))
        for name in ("loop_detector_enabled", "global_optimization_enabled", "loop_ba_running"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool.")
        if self.global_optimization_enabled and not self.loop_detector_enabled:
            raise ReconstructionValidationError(
                "Native stella global optimization cannot be enabled without loop closure."
            )
        object.__setattr__(self, "map_revision", require_nonnegative_int(self.map_revision, field_name="map_revision"))
        object.__setattr__(
            self,
            "optimized_pose_revision",
            require_positive_int(self.optimized_pose_revision, field_name="optimized_pose_revision"),
        )
        object.__setattr__(self, "keyframe_count", require_nonnegative_int(self.keyframe_count, field_name="keyframe_count"))
        object.__setattr__(self, "landmark_count", require_nonnegative_int(self.landmark_count, field_name="landmark_count"))
        object.__setattr__(
            self,
            "accepted_loop_closure_requests",
            require_nonnegative_int(
                self.accepted_loop_closure_requests,
                field_name="accepted_loop_closure_requests",
            ),
        )
        object.__setattr__(
            self,
            "relocalization_count",
            require_nonnegative_int(self.relocalization_count, field_name="relocalization_count"),
        )
        object.__setattr__(self, "last_relocalization_frame_id", str(self.last_relocalization_frame_id).strip())
        poses = tuple(self.optimized_keyframe_poses)
        if any(not isinstance(item, OptimizedKeyframePose) for item in poses):
            raise TypeError("optimized_keyframe_poses must contain OptimizedKeyframePose values.")
        if len({item.keyframe_id for item in poses}) != len(poses):
            raise ReconstructionValidationError("Native optimized keyframe IDs must be unique.")
        if any(item.optimization_revision != self.optimized_pose_revision for item in poses):
            raise ReconstructionValidationError(
                "Native optimized keyframe poses must use optimized_pose_revision."
            )
        if len(poses) != self.keyframe_count:
            raise ReconstructionValidationError(
                "Native optimized keyframe pose count must match keyframe_count."
            )
        object.__setattr__(self, "optimized_keyframe_poses", poses)
        gauge = self.gauge
        if not isinstance(gauge, ReconstructionGauge):
            raise TypeError("gauge must be ReconstructionGauge.")
        if (
            gauge.scale_status is not SfmScaleStatus.ARBITRARY
            or gauge.alignment_status is not SfmWorldAlignmentStatus.ARBITRARY
            or gauge.unit_to_metre is not None
        ):
            raise ReconstructionValidationError(
                "Native monocular global-consistency status must remain arbitrary-scale."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "loop_detector_enabled": self.loop_detector_enabled,
            "global_optimization_enabled": self.global_optimization_enabled,
            "loop_ba_running": self.loop_ba_running,
            "map_revision": self.map_revision,
            "optimized_pose_revision": self.optimized_pose_revision,
            "keyframe_count": self.keyframe_count,
            "landmark_count": self.landmark_count,
            "accepted_loop_closure_requests": self.accepted_loop_closure_requests,
            "relocalization_count": self.relocalization_count,
            "last_relocalization_frame_id": self.last_relocalization_frame_id,
            "optimized_keyframe_poses": [item.as_dict() for item in self.optimized_keyframe_poses],
            "gauge": self.gauge.as_dict(),
        }


def native_loop_closure_result_from_payload(
    payload: Mapping[str, object], *, expected_session_id: str
) -> NativeLoopClosureRequestResult:
    try:
        return NativeLoopClosureRequestResult(
            session_id=expected_session_id,
            from_keyframe_id=str(payload["from_keyframe_id"]),
            to_keyframe_id=str(payload["to_keyframe_id"]),
            accepted=payload["accepted"],
            accepted_request_count=int(payload["accepted_request_count"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ReconstructionValidationError):
            raise
        raise LiveSlamProtocolError("Invalid native loop-closure request result payload.") from exc


def native_global_consistency_status_from_payload(
    payload: Mapping[str, object], *, expected_session_id: str
) -> NativeGlobalConsistencyStatus:
    raw_poses = payload.get("optimized_keyframe_poses")
    if not isinstance(raw_poses, list):
        raise LiveSlamProtocolError("optimized_keyframe_poses must be a JSON array.")
    try:
        revision = int(payload["optimized_pose_revision"])
        poses: list[OptimizedKeyframePose] = []
        for item in raw_poses:
            if not isinstance(item, Mapping):
                raise LiveSlamProtocolError("optimized keyframe pose must be an object.")
            if item.get("pose_convention") != "T_reconstruction_from_camera":
                raise LiveSlamProtocolError(
                    "Native optimized keyframe pose must use T_reconstruction_from_camera."
                )
            poses.append(
                OptimizedKeyframePose(
                    keyframe_id=str(item["keyframe_id"]),
                    frame_id=str(item["frame_id"]),
                    timestamp_ns=int(item["timestamp_ns"]),
                    source_pose_id=str(item["source_pose_id"]),
                    optimization_revision=revision,
                    T_reconstruction_from_camera=np.asarray(
                        item["T_reconstruction_from_camera"], dtype=np.float64
                    ),
                )
            )
        return NativeGlobalConsistencyStatus(
            session_id=expected_session_id,
            loop_detector_enabled=payload["loop_detector_enabled"],
            global_optimization_enabled=payload["global_optimization_enabled"],
            loop_ba_running=payload["loop_ba_running"],
            map_revision=int(payload["map_revision"]),
            optimized_pose_revision=revision,
            keyframe_count=int(payload["keyframe_count"]),
            landmark_count=int(payload["landmark_count"]),
            accepted_loop_closure_requests=int(payload["accepted_loop_closure_requests"]),
            relocalization_count=int(payload["relocalization_count"]),
            last_relocalization_frame_id=str(payload.get("last_relocalization_frame_id", "")),
            optimized_keyframe_poses=tuple(poses),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, (LiveSlamProtocolError, ReconstructionValidationError)):
            raise
        raise LiveSlamProtocolError("Invalid native global-consistency status payload.") from exc
