"""Relocalization policy helpers for Phase 8.5A."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.reconstruction.global_consistency import RelocalizationResult, RelocalizationStatus
from core.reconstruction.sfm_coordinates import validate_rigid_matrix
from core.reconstruction.validation import require_nonnegative_int, require_probability


@dataclass(frozen=True, slots=True)
class RelocalizationAcceptancePolicy:
    minimum_confidence: float = 0.70
    minimum_inliers: int = 30

    def __post_init__(self) -> None:
        object.__setattr__(self, "minimum_confidence", require_probability(self.minimum_confidence, field_name="minimum_confidence"))
        object.__setattr__(self, "minimum_inliers", require_nonnegative_int(self.minimum_inliers, field_name="minimum_inliers"))

    def evaluate(
        self,
        *,
        query_frame_id: str,
        map_revision: int,
        matched_keyframe_id: str,
        confidence: float,
        inlier_count: int,
        T_reconstruction_from_camera: np.ndarray | None,
    ) -> RelocalizationResult:
        confidence = require_probability(confidence, field_name="confidence")
        inlier_count = require_nonnegative_int(inlier_count, field_name="inlier_count")
        if confidence < self.minimum_confidence:
            return RelocalizationResult(
                query_frame_id=query_frame_id,
                map_revision=map_revision,
                status=RelocalizationStatus.REJECTED,
                confidence=confidence,
                inlier_count=inlier_count,
                reason="confidence_below_threshold",
            )
        if inlier_count < self.minimum_inliers:
            return RelocalizationResult(
                query_frame_id=query_frame_id,
                map_revision=map_revision,
                status=RelocalizationStatus.INSUFFICIENT_MATCHES,
                confidence=confidence,
                inlier_count=inlier_count,
                reason="inliers_below_threshold",
            )
        if T_reconstruction_from_camera is None:
            return RelocalizationResult(
                query_frame_id=query_frame_id,
                map_revision=map_revision,
                status=RelocalizationStatus.FAILED,
                confidence=confidence,
                inlier_count=inlier_count,
                reason="pose_unavailable",
            )
        pose = validate_rigid_matrix(
            T_reconstruction_from_camera,
            field_name="T_reconstruction_from_camera",
        )
        return RelocalizationResult(
            query_frame_id=query_frame_id,
            map_revision=map_revision,
            status=RelocalizationStatus.SUCCESS,
            confidence=confidence,
            inlier_count=inlier_count,
            matched_keyframe_id=matched_keyframe_id,
            T_reconstruction_from_camera=pose,
            reason="accepted",
        )
