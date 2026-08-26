"""I4.1.6 additive decoupled global/local dense-candidate integration.

This module turns the I4.1.5 counterfactual into an executable, opt-in
localization path without changing the production default.  Global PyCOLMAP
SfM remains the single geometry authority.  Additional dense candidates are
source observations only; every non-anchor candidate must be localized into
that existing reconstruction frame before it can become eligible for DA3,
dense consistency, or fusion.

I4.1.6A deliberately stops at real candidate localization.  It does not run
DA3, dense pair qualification, fusion, filtering, persistence, metric-scale
promotion, measurement, or mesh.  The subsequent I4.1.6B validation may consume
its saved evidence if localization is sufficiently strong.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import floor, isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.sfm_contracts import (
    OfflineSfmResult,
    SfmCameraEstimate,
    SfmCameraModelEstimate,
)
from core.reconstruction.sfm_coordinates import invert_rigid_matrix

I4_1_6_SCHEMA_VERSION = 1
_EPS = 1e-12


class DenseCandidatePlannerMode(str, Enum):
    GLOBAL_ANCHORS_ONLY = "global_anchors_only"
    GLOBAL_ANCHORS_PLUS_LOCAL_NEIGHBORS = "global_anchors_plus_local_neighbors"
    UNIFORM_LOCALIZED_DENSE_POOL = "uniform_localized_dense_pool"


class DenseCandidateLocalizationStatus(str, Enum):
    GLOBAL_ANCHOR_REUSED = "global_anchor_reused"
    LOCALIZED = "localized"
    REJECTED_INSUFFICIENT_MATCHES = "rejected_insufficient_matches"
    REJECTED_POSE_ESTIMATION = "rejected_pose_estimation"
    REJECTED_INLIER_GATE = "rejected_inlier_gate"
    REJECTED_REPROJECTION_GATE = "rejected_reprojection_gate"


@dataclass(frozen=True, slots=True)
class DenseCandidateLocalizationPolicy:
    nearest_anchor_count: int = 4
    sift_features: int = 6000
    maximum_landmarks_per_anchor: int = 5000
    descriptor_ratio_test: float = 0.78
    minimum_correspondences: int = 40
    minimum_inliers: int = 30
    minimum_inlier_fraction: float = 0.35
    ransac_reprojection_error_px: float = 6.0
    maximum_median_reprojection_error_px: float = 3.0
    maximum_p90_reprojection_error_px: float = 8.0

    def __post_init__(self) -> None:
        for name in (
            "nearest_anchor_count",
            "sift_features",
            "maximum_landmarks_per_anchor",
            "minimum_correspondences",
            "minimum_inliers",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ReconstructionValidationError(f"{name} must be a positive integer.")
            object.__setattr__(self, name, int(value))
        for name in (
            "descriptor_ratio_test",
            "minimum_inlier_fraction",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 < value < 1.0:
                raise ReconstructionValidationError(f"{name} must be finite and in (0, 1).")
            object.__setattr__(self, name, value)
        for name in (
            "ransac_reprojection_error_px",
            "maximum_median_reprojection_error_px",
            "maximum_p90_reprojection_error_px",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ReconstructionValidationError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        if self.minimum_inliers > self.minimum_correspondences:
            raise ReconstructionValidationError(
                "minimum_inliers cannot exceed minimum_correspondences."
            )
        if self.maximum_median_reprojection_error_px > self.maximum_p90_reprojection_error_px:
            raise ReconstructionValidationError(
                "median reprojection gate cannot exceed the p90 reprojection gate."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "nearest_anchor_count": self.nearest_anchor_count,
            "sift_features": self.sift_features,
            "maximum_landmarks_per_anchor": self.maximum_landmarks_per_anchor,
            "descriptor_ratio_test": self.descriptor_ratio_test,
            "minimum_correspondences": self.minimum_correspondences,
            "minimum_inliers": self.minimum_inliers,
            "minimum_inlier_fraction": self.minimum_inlier_fraction,
            "ransac_reprojection_error_px": self.ransac_reprojection_error_px,
            "maximum_median_reprojection_error_px": self.maximum_median_reprojection_error_px,
            "maximum_p90_reprojection_error_px": self.maximum_p90_reprojection_error_px,
        }


@dataclass(frozen=True, slots=True)
class DenseCandidateSpec:
    source_sequence_index: int
    timestamp_ns: int
    is_global_anchor: bool
    global_workspace_order: int | None = None
    global_keyframe_id: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.source_sequence_index, bool) or int(self.source_sequence_index) < 0:
            raise ReconstructionValidationError("source_sequence_index must be non-negative.")
        if isinstance(self.timestamp_ns, bool) or int(self.timestamp_ns) < 0:
            raise ReconstructionValidationError("timestamp_ns must be non-negative.")
        object.__setattr__(self, "source_sequence_index", int(self.source_sequence_index))
        object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))
        if self.is_global_anchor:
            if self.global_workspace_order is None or int(self.global_workspace_order) < 0:
                raise ReconstructionValidationError(
                    "Global-anchor candidates require global_workspace_order."
                )
            if not str(self.global_keyframe_id).strip():
                raise ReconstructionValidationError(
                    "Global-anchor candidates require global_keyframe_id."
                )
            object.__setattr__(self, "global_workspace_order", int(self.global_workspace_order))
            object.__setattr__(self, "global_keyframe_id", str(self.global_keyframe_id).strip())
        else:
            if self.global_workspace_order is not None or str(self.global_keyframe_id).strip():
                raise ReconstructionValidationError(
                    "Non-anchor candidates must not carry global-anchor identity."
                )
            object.__setattr__(self, "global_keyframe_id", "")

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_sequence_index": self.source_sequence_index,
            "timestamp_ns": self.timestamp_ns,
            "is_global_anchor": self.is_global_anchor,
            "global_workspace_order": self.global_workspace_order,
            "global_keyframe_id": self.global_keyframe_id or None,
        }


@dataclass(frozen=True, slots=True)
class DenseCandidatePlan:
    mode: DenseCandidatePlannerMode
    source_total_frames: int
    source_fps: float
    source_duration_seconds: float
    global_anchor_count: int
    dense_candidate_cap: int
    local_target_fps: float
    candidates: tuple[DenseCandidateSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.mode, DenseCandidatePlannerMode):
            raise TypeError("mode must be DenseCandidatePlannerMode.")
        if isinstance(self.source_total_frames, bool) or int(self.source_total_frames) < 2:
            raise ReconstructionValidationError("source_total_frames must be at least 2.")
        fps = float(self.source_fps)
        duration = float(self.source_duration_seconds)
        target = float(self.local_target_fps)
        if not isfinite(fps) or fps <= 0.0 or not isfinite(duration) or duration <= 0.0:
            raise ReconstructionValidationError("source FPS/duration must be finite and positive.")
        if not isfinite(target) or target <= 0.0:
            raise ReconstructionValidationError("local_target_fps must be finite and positive.")
        if isinstance(self.dense_candidate_cap, bool) or int(self.dense_candidate_cap) < 2:
            raise ReconstructionValidationError("dense_candidate_cap must be at least 2.")
        candidates = tuple(self.candidates)
        if not candidates:
            raise ReconstructionValidationError("Dense candidate plan must not be empty.")
        indices = [item.source_sequence_index for item in candidates]
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise ReconstructionValidationError(
                "Dense candidate source indices must be unique and strictly increasing."
            )
        if len(candidates) > int(self.dense_candidate_cap):
            raise ReconstructionValidationError("Dense candidate plan exceeds its cap.")
        if indices[0] < 0 or indices[-1] >= int(self.source_total_frames):
            raise ReconstructionValidationError("Dense candidate plan references frames outside the source.")
        object.__setattr__(self, "source_total_frames", int(self.source_total_frames))
        object.__setattr__(self, "source_fps", fps)
        object.__setattr__(self, "source_duration_seconds", duration)
        object.__setattr__(self, "dense_candidate_cap", int(self.dense_candidate_cap))
        object.__setattr__(self, "local_target_fps", target)
        object.__setattr__(self, "candidates", candidates)

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def extra_candidate_count(self) -> int:
        return sum(not item.is_global_anchor for item in self.candidates)

    @property
    def source_span_fraction(self) -> float:
        if len(self.candidates) < 2:
            return 0.0
        span = self.candidates[-1].source_sequence_index - self.candidates[0].source_sequence_index
        return float(span / max(1, self.source_total_frames - 1))

    def as_dict(self) -> dict[str, Any]:
        indices = np.asarray([item.source_sequence_index for item in self.candidates], dtype=np.float64)
        gaps_s = np.diff(indices) / self.source_fps if len(indices) >= 2 else np.empty((0,), dtype=np.float64)
        return {
            "mode": self.mode.value,
            "source_total_frames": self.source_total_frames,
            "source_fps": self.source_fps,
            "source_duration_seconds": self.source_duration_seconds,
            "global_anchor_count": self.global_anchor_count,
            "dense_candidate_cap": self.dense_candidate_cap,
            "local_target_fps": self.local_target_fps,
            "candidate_count": self.candidate_count,
            "extra_candidate_count": self.extra_candidate_count,
            "source_span_fraction": self.source_span_fraction,
            "adjacent_gap_seconds": {
                "p50": None if gaps_s.size == 0 else float(np.quantile(gaps_s, 0.50)),
                "p90": None if gaps_s.size == 0 else float(np.quantile(gaps_s, 0.90)),
                "max": None if gaps_s.size == 0 else float(np.max(gaps_s)),
                "fraction_le_0_75": None if gaps_s.size == 0 else float(np.mean(gaps_s <= 0.75 + 1e-12)),
            },
            "candidates": [item.as_dict() for item in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class PoseEstimationEvidence:
    accepted: bool
    T_reconstruction_from_camera: np.ndarray | None
    correspondence_count: int
    inlier_count: int
    inlier_fraction: float
    median_reprojection_error_px: float | None
    p90_reprojection_error_px: float | None
    inlier_mask: np.ndarray
    rejection_status: DenseCandidateLocalizationStatus | None

    def __post_init__(self) -> None:
        mask = np.asarray(self.inlier_mask, dtype=bool).reshape(-1).copy()
        mask.setflags(write=False)
        object.__setattr__(self, "inlier_mask", mask)
        if self.accepted:
            matrix = np.asarray(self.T_reconstruction_from_camera, dtype=np.float64)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ReconstructionValidationError("Accepted pose evidence requires a finite 4x4 pose.")
            matrix = np.ascontiguousarray(matrix).copy(); matrix.setflags(write=False)
            object.__setattr__(self, "T_reconstruction_from_camera", matrix)
            if self.rejection_status is not None:
                raise ReconstructionValidationError("Accepted pose evidence cannot carry a rejection status.")
        else:
            if self.T_reconstruction_from_camera is not None:
                raise ReconstructionValidationError("Rejected pose evidence must not carry a pose.")
            if self.rejection_status is None:
                raise ReconstructionValidationError("Rejected pose evidence requires a rejection status.")


@dataclass(frozen=True, slots=True)
class DenseCandidateLocalizationResult:
    source_sequence_index: int
    timestamp_ns: int
    status: DenseCandidateLocalizationStatus
    accepted: bool
    is_global_anchor: bool
    global_workspace_order: int | None
    keyframe_id: str
    T_reconstruction_from_camera: np.ndarray | None
    correspondence_count: int
    inlier_count: int
    inlier_fraction: float
    median_reprojection_error_px: float | None
    p90_reprojection_error_px: float | None
    anchor_workspace_orders_used: tuple[int, ...]
    matched_landmark_ids: tuple[str, ...] = ()
    matched_feature_xy: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if self.accepted and self.T_reconstruction_from_camera is None:
            raise ReconstructionValidationError("Accepted localization requires a pose.")
        if not self.accepted and self.T_reconstruction_from_camera is not None:
            raise ReconstructionValidationError("Rejected localization must not carry a pose.")
        if len(self.matched_landmark_ids) != len(self.matched_feature_xy):
            raise ReconstructionValidationError("Matched landmark IDs and feature coordinates must align.")
        if self.accepted:
            matrix = np.asarray(self.T_reconstruction_from_camera, dtype=np.float64)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ReconstructionValidationError("Localization pose must be finite 4x4.")
            matrix = np.ascontiguousarray(matrix).copy(); matrix.setflags(write=False)
            object.__setattr__(self, "T_reconstruction_from_camera", matrix)
        object.__setattr__(self, "matched_landmark_ids", tuple(str(v) for v in self.matched_landmark_ids))
        object.__setattr__(
            self,
            "matched_feature_xy",
            tuple((float(x), float(y)) for x, y in self.matched_feature_xy),
        )
        object.__setattr__(self, "anchor_workspace_orders_used", tuple(int(v) for v in self.anchor_workspace_orders_used))

    def as_dict(self, *, include_matches: bool = False) -> dict[str, Any]:
        payload = {
            "source_sequence_index": int(self.source_sequence_index),
            "timestamp_ns": int(self.timestamp_ns),
            "status": self.status.value,
            "accepted": bool(self.accepted),
            "is_global_anchor": bool(self.is_global_anchor),
            "global_workspace_order": self.global_workspace_order,
            "keyframe_id": self.keyframe_id,
            "T_reconstruction_from_camera": (
                None if self.T_reconstruction_from_camera is None else self.T_reconstruction_from_camera.tolist()
            ),
            "correspondence_count": int(self.correspondence_count),
            "inlier_count": int(self.inlier_count),
            "inlier_fraction": float(self.inlier_fraction),
            "median_reprojection_error_px": self.median_reprojection_error_px,
            "p90_reprojection_error_px": self.p90_reprojection_error_px,
            "anchor_workspace_orders_used": list(self.anchor_workspace_orders_used),
            "matched_inlier_count": len(self.matched_landmark_ids),
        }
        if include_matches:
            payload["matched_landmark_ids"] = list(self.matched_landmark_ids)
            payload["matched_feature_xy"] = [list(v) for v in self.matched_feature_xy]
        return payload


def probe_video_metadata(video_path: str | Path) -> tuple[int, float, float]:
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise ReconstructionValidationError(f"I4.1.6 video is missing: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ReconstructionValidationError(f"I4.1.6 could not open video: {path}")
    try:
        total = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if total < 2 or not isfinite(fps) or fps <= 0.0:
        raise ReconstructionValidationError("I4.1.6 video metadata is invalid.")
    duration = float((total - 1) / fps)
    return total, fps, duration


def _uniform_indices(total_frames: int, count: int) -> tuple[int, ...]:
    if total_frames < 2 or count < 2:
        raise ReconstructionValidationError("Uniform sampling requires total_frames/count >= 2.")
    count = min(total_frames, count)
    values = tuple(
        int(round(i * (total_frames - 1) / (count - 1)))
        for i in range(count)
    )
    if len(values) != len(set(values)):
        raise ReconstructionValidationError("Uniform dense-candidate schedule produced duplicates.")
    return values


def _bounded_even_subset(indices: Iterable[int], cap: int) -> tuple[int, ...]:
    values = tuple(sorted(set(int(v) for v in indices)))
    if len(values) <= cap:
        return values
    selected = _uniform_indices(len(values), cap)
    return tuple(values[i] for i in selected)


def build_dense_candidate_plan(
    sfm_result: OfflineSfmResult,
    *,
    source_total_frames: int,
    source_fps: float,
    mode: DenseCandidatePlannerMode,
    dense_candidate_cap: int = 120,
    local_target_fps: float = 2.0,
) -> DenseCandidatePlan:
    if not isinstance(sfm_result, OfflineSfmResult):
        raise TypeError("sfm_result must be OfflineSfmResult.")
    if not isinstance(mode, DenseCandidatePlannerMode):
        raise TypeError("mode must be DenseCandidatePlannerMode.")
    total = int(source_total_frames)
    fps = float(source_fps)
    cap = int(dense_candidate_cap)
    target = float(local_target_fps)
    if total < 2 or cap < 2 or not isfinite(fps) or fps <= 0.0 or not isfinite(target) or target <= 0.0:
        raise ReconstructionValidationError("I4.1.6 candidate-plan inputs are invalid.")

    registered = {item.keyframe_id: item for item in sfm_result.cameras}
    anchors = sorted(
        (item for item in sfm_result.workspace.images if item.keyframe_id in registered),
        key=lambda item: (int(item.source_sequence_index), int(item.order)),
    )
    if len(anchors) < 2:
        raise ReconstructionValidationError("I4.1.6 requires at least two registered global SfM anchors.")
    anchor_by_index = {int(item.source_sequence_index): item for item in anchors}
    anchor_indices = tuple(sorted(anchor_by_index))

    if mode is DenseCandidatePlannerMode.GLOBAL_ANCHORS_ONLY:
        indices = _bounded_even_subset(anchor_indices, cap)
    elif mode is DenseCandidatePlannerMode.GLOBAL_ANCHORS_PLUS_LOCAL_NEIGHBORS:
        step = max(1, int(round(fps / target)))
        raw: list[int] = []
        for index in anchor_indices:
            raw.extend((max(0, index - step), index, min(total - 1, index + step)))
        indices = _bounded_even_subset(raw, cap)
    else:
        duration = (total - 1) / fps
        nominal = min(total, int(floor(duration * target + 1e-12)) + 1)
        count = min(cap, max(2, nominal))
        indices = _uniform_indices(total, count)

    candidates: list[DenseCandidateSpec] = []
    for index in indices:
        anchor = anchor_by_index.get(index)
        candidates.append(
            DenseCandidateSpec(
                source_sequence_index=index,
                timestamp_ns=int(round(index / fps * 1_000_000_000)),
                is_global_anchor=anchor is not None,
                global_workspace_order=None if anchor is None else int(anchor.order),
                global_keyframe_id="" if anchor is None else str(anchor.keyframe_id),
            )
        )
    return DenseCandidatePlan(
        mode=mode,
        source_total_frames=total,
        source_fps=fps,
        source_duration_seconds=(total - 1) / fps,
        global_anchor_count=len(anchors),
        dense_candidate_cap=cap,
        local_target_fps=target,
        candidates=tuple(candidates),
    )


def _camera_matrix_and_distortion(
    model: SfmCameraModelEstimate,
    *,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    name = model.backend_model_name.strip().upper()
    params = tuple(float(v) for v in model.parameters)
    if name == "SIMPLE_PINHOLE" and len(params) == 3:
        f, cx, cy = params; fx = fy = f; dist = np.zeros((5, 1), dtype=np.float64)
    elif name == "PINHOLE" and len(params) == 4:
        fx, fy, cx, cy = params; dist = np.zeros((5, 1), dtype=np.float64)
    elif name == "SIMPLE_RADIAL" and len(params) == 4:
        f, cx, cy, k1 = params; fx = fy = f; dist = np.asarray([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64).reshape(-1, 1)
    elif name == "RADIAL" and len(params) == 5:
        f, cx, cy, k1, k2 = params; fx = fy = f; dist = np.asarray([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64).reshape(-1, 1)
    elif name == "OPENCV" and len(params) == 8:
        fx, fy, cx, cy, k1, k2, p1, p2 = params; dist = np.asarray([k1, k2, p1, p2], dtype=np.float64).reshape(-1, 1)
    else:
        raise ReconstructionValidationError(
            f"I4.1.6 localization does not support camera model {name!r}/{len(params)}."
        )
    sx = float(image_width) / float(model.image_size.width)
    sy = float(image_height) / float(model.image_size.height)
    K = np.asarray(
        [[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return K, dist


def estimate_pose_from_correspondences(
    points2d: object,
    points3d: object,
    *,
    camera_model: SfmCameraModelEstimate,
    image_width: int,
    image_height: int,
    policy: DenseCandidateLocalizationPolicy,
) -> PoseEstimationEvidence:
    if not isinstance(camera_model, SfmCameraModelEstimate):
        raise TypeError("camera_model must be SfmCameraModelEstimate.")
    if not isinstance(policy, DenseCandidateLocalizationPolicy):
        raise TypeError("policy must be DenseCandidateLocalizationPolicy.")
    p2 = np.asarray(points2d, dtype=np.float64)
    p3 = np.asarray(points3d, dtype=np.float64)
    if p2.ndim != 2 or p2.shape[1] != 2 or p3.ndim != 2 or p3.shape[1] != 3 or len(p2) != len(p3):
        raise ReconstructionValidationError("I4.1.6 PnP correspondences must be Nx2/Nx3 and aligned.")
    if not np.isfinite(p2).all() or not np.isfinite(p3).all():
        raise ReconstructionValidationError("I4.1.6 PnP correspondences must be finite.")
    count = int(len(p2))
    empty_mask = np.zeros((count,), dtype=bool)
    if count < policy.minimum_correspondences:
        return PoseEstimationEvidence(
            accepted=False, T_reconstruction_from_camera=None,
            correspondence_count=count, inlier_count=0, inlier_fraction=0.0,
            median_reprojection_error_px=None, p90_reprojection_error_px=None,
            inlier_mask=empty_mask,
            rejection_status=DenseCandidateLocalizationStatus.REJECTED_INSUFFICIENT_MATCHES,
        )
    K, dist = _camera_matrix_and_distortion(
        camera_model, image_width=int(image_width), image_height=int(image_height)
    )
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        p3, p2, K, dist,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=float(policy.ransac_reprojection_error_px),
        confidence=0.999,
        iterationsCount=2000,
    )
    if not success or rvec is None or tvec is None or inliers is None:
        return PoseEstimationEvidence(
            accepted=False, T_reconstruction_from_camera=None,
            correspondence_count=count, inlier_count=0, inlier_fraction=0.0,
            median_reprojection_error_px=None, p90_reprojection_error_px=None,
            inlier_mask=empty_mask,
            rejection_status=DenseCandidateLocalizationStatus.REJECTED_POSE_ESTIMATION,
        )
    mask = np.zeros((count,), dtype=bool)
    mask[np.asarray(inliers, dtype=np.int64).reshape(-1)] = True
    inlier_count = int(mask.sum())
    inlier_fraction = float(inlier_count / max(1, count))
    if inlier_count < policy.minimum_inliers or inlier_fraction < policy.minimum_inlier_fraction:
        return PoseEstimationEvidence(
            accepted=False, T_reconstruction_from_camera=None,
            correspondence_count=count, inlier_count=inlier_count, inlier_fraction=inlier_fraction,
            median_reprojection_error_px=None, p90_reprojection_error_px=None,
            inlier_mask=mask,
            rejection_status=DenseCandidateLocalizationStatus.REJECTED_INLIER_GATE,
        )
    # Refine against RANSAC inliers while keeping the SfM-estimated camera fixed.
    try:
        cv2.solvePnPRefineLM(p3[mask], p2[mask], K, dist, rvec, tvec)
    except cv2.error:
        pass
    projected, _ = cv2.projectPoints(p3[mask], rvec, tvec, K, dist)
    residuals = np.linalg.norm(projected.reshape(-1, 2) - p2[mask], axis=1)
    median = float(np.median(residuals))
    p90 = float(np.quantile(residuals, 0.90))
    if median > policy.maximum_median_reprojection_error_px or p90 > policy.maximum_p90_reprojection_error_px:
        return PoseEstimationEvidence(
            accepted=False, T_reconstruction_from_camera=None,
            correspondence_count=count, inlier_count=inlier_count, inlier_fraction=inlier_fraction,
            median_reprojection_error_px=median, p90_reprojection_error_px=p90,
            inlier_mask=mask,
            rejection_status=DenseCandidateLocalizationStatus.REJECTED_REPROJECTION_GATE,
        )
    rotation, _ = cv2.Rodrigues(rvec)
    T_camera_from_reconstruction = np.eye(4, dtype=np.float64)
    T_camera_from_reconstruction[:3, :3] = np.asarray(rotation, dtype=np.float64)
    T_camera_from_reconstruction[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    T_reconstruction_from_camera = invert_rigid_matrix(T_camera_from_reconstruction)
    return PoseEstimationEvidence(
        accepted=True,
        T_reconstruction_from_camera=T_reconstruction_from_camera,
        correspondence_count=count,
        inlier_count=inlier_count,
        inlier_fraction=inlier_fraction,
        median_reprojection_error_px=median,
        p90_reprojection_error_px=p90,
        inlier_mask=mask,
        rejection_status=None,
    )


@dataclass(slots=True)
class _AnchorDescriptorRecord:
    workspace_order: int
    source_sequence_index: int
    keyframe_id: str
    descriptors: np.ndarray
    landmark_ids: tuple[str, ...]
    points3d: np.ndarray


class SfmAnchorDescriptorIndex:
    """Descriptor index built only from landmarks already authorized by global SfM."""

    def __init__(self, sfm_result: OfflineSfmResult, policy: DenseCandidateLocalizationPolicy) -> None:
        if not isinstance(sfm_result, OfflineSfmResult):
            raise TypeError("sfm_result must be OfflineSfmResult.")
        self.sfm_result = sfm_result
        self.policy = policy
        self._sift = cv2.SIFT_create(nfeatures=int(policy.sift_features))
        self._records = self._build_records()

    @property
    def records(self) -> tuple[_AnchorDescriptorRecord, ...]:
        return self._records

    def _build_records(self) -> tuple[_AnchorDescriptorRecord, ...]:
        root = Path(self.sfm_result.workspace.workspace_root).expanduser().resolve()
        registered = {item.keyframe_id for item in self.sfm_result.cameras}
        observations_by_keyframe: dict[str, list[tuple[str, tuple[float, float], np.ndarray, int]]] = {}
        for landmark in self.sfm_result.landmarks:
            track_len = len(landmark.observations)
            for obs in landmark.observations:
                if obs.keyframe_id in registered:
                    observations_by_keyframe.setdefault(obs.keyframe_id, []).append(
                        (landmark.landmark_id, obs.feature_xy, landmark.position_reconstruction, track_len)
                    )
        output: list[_AnchorDescriptorRecord] = []
        for workspace in self.sfm_result.workspace.images:
            if workspace.keyframe_id not in registered:
                continue
            source = observations_by_keyframe.get(workspace.keyframe_id, [])
            if not source:
                continue
            source = sorted(source, key=lambda item: (-int(item[3]), item[0]))[: self.policy.maximum_landmarks_per_anchor]
            image_path = root / workspace.staged_image_path
            gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ReconstructionValidationError(
                    f"I4.1.6 could not decode global anchor image: {image_path}"
                )
            sx = float(gray.shape[1]) / float(workspace.image_size.width)
            sy = float(gray.shape[0]) / float(workspace.image_size.height)
            # Detect descriptors with the same OpenCV SIFT implementation used for
            # query candidates, then associate them to the nearest authoritative
            # COLMAP observation.  This is more stable than inventing scale/orientation
            # for a descriptor at an arbitrary observation coordinate.
            detected, descriptors = self._sift.detectAndCompute(gray, None)
            if descriptors is None or not detected:
                continue
            cell_size = 4.0
            maximum_distance = 4.0
            grid: dict[tuple[int, int], list[int]] = {}
            scaled_xy: list[tuple[float, float]] = []
            for index, (_, xy, _, _) in enumerate(source):
                point = (float(xy[0] * sx), float(xy[1] * sy))
                scaled_xy.append(point)
                cell = (int(floor(point[0] / cell_size)), int(floor(point[1] / cell_size)))
                grid.setdefault(cell, []).append(index)
            best_by_landmark: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
            for kp, descriptor in zip(detected, descriptors):
                x, y = float(kp.pt[0]), float(kp.pt[1])
                cx, cy = int(floor(x / cell_size)), int(floor(y / cell_size))
                best_index = None
                best_distance = float("inf")
                for gx in (cx - 1, cx, cx + 1):
                    for gy in (cy - 1, cy, cy + 1):
                        for candidate_index in grid.get((gx, gy), ()):
                            ox, oy = scaled_xy[candidate_index]
                            distance = float(np.hypot(x - ox, y - oy))
                            if distance < best_distance:
                                best_distance = distance
                                best_index = candidate_index
                if best_index is None or best_distance > maximum_distance:
                    continue
                landmark_id, _, xyz, _ = source[best_index]
                current = best_by_landmark.get(str(landmark_id))
                candidate = (best_distance, np.asarray(descriptor, dtype=np.float32), np.asarray(xyz, dtype=np.float64))
                if current is None or candidate[0] < current[0]:
                    best_by_landmark[str(landmark_id)] = candidate
            if not best_by_landmark:
                continue
            ordered = sorted(best_by_landmark.items(), key=lambda item: item[0])
            output.append(
                _AnchorDescriptorRecord(
                    workspace_order=int(workspace.order),
                    source_sequence_index=int(workspace.source_sequence_index),
                    keyframe_id=str(workspace.keyframe_id),
                    descriptors=np.ascontiguousarray(np.vstack([item[1][1] for item in ordered]), dtype=np.float32),
                    landmark_ids=tuple(item[0] for item in ordered),
                    points3d=np.ascontiguousarray(np.vstack([item[1][2] for item in ordered]), dtype=np.float64),
                )
            )
        if len(output) < 2:
            raise ReconstructionValidationError(
                "I4.1.6 could not build descriptors for at least two registered SfM anchors."
            )
        return tuple(sorted(output, key=lambda item: (item.source_sequence_index, item.workspace_order)))

    def _nearest_records(self, source_sequence_index: int) -> tuple[_AnchorDescriptorRecord, ...]:
        ranked = sorted(
            self._records,
            key=lambda item: (abs(item.source_sequence_index - int(source_sequence_index)), item.workspace_order),
        )
        return tuple(ranked[: self.policy.nearest_anchor_count])

    def localize(
        self,
        *,
        source_sequence_index: int,
        timestamp_ns: int,
        image_path: str | Path,
        camera_model: SfmCameraModelEstimate,
    ) -> DenseCandidateLocalizationResult:
        path = Path(image_path).expanduser().resolve()
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ReconstructionValidationError(f"I4.1.6 could not decode dense candidate: {path}")
        keypoints, descriptors = self._sift.detectAndCompute(gray, None)
        anchors = self._nearest_records(int(source_sequence_index))
        anchor_orders = tuple(item.workspace_order for item in anchors)
        keyframe_id = f"i4-local-{int(source_sequence_index):08d}"
        if descriptors is None or not keypoints:
            return DenseCandidateLocalizationResult(
                source_sequence_index=int(source_sequence_index), timestamp_ns=int(timestamp_ns),
                status=DenseCandidateLocalizationStatus.REJECTED_INSUFFICIENT_MATCHES,
                accepted=False, is_global_anchor=False, global_workspace_order=None,
                keyframe_id=keyframe_id, T_reconstruction_from_camera=None,
                correspondence_count=0, inlier_count=0, inlier_fraction=0.0,
                median_reprojection_error_px=None, p90_reprojection_error_px=None,
                anchor_workspace_orders_used=anchor_orders,
            )
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        # candidate feature -> (distance, landmark_id, xyz)
        best_by_candidate: dict[int, tuple[float, str, np.ndarray]] = {}
        for anchor in anchors:
            if len(anchor.descriptors) < 2:
                continue
            pairs = matcher.knnMatch(
                np.ascontiguousarray(descriptors, dtype=np.float32), anchor.descriptors, k=2
            )
            for pair in pairs:
                if len(pair) != 2:
                    continue
                first, second = pair
                if float(first.distance) >= self.policy.descriptor_ratio_test * float(second.distance):
                    continue
                idx = int(first.trainIdx)
                if not 0 <= idx < len(anchor.landmark_ids):
                    continue
                candidate_idx = int(first.queryIdx)
                candidate = (
                    float(first.distance),
                    anchor.landmark_ids[idx],
                    np.asarray(anchor.points3d[idx], dtype=np.float64),
                )
                existing = best_by_candidate.get(candidate_idx)
                if existing is None or candidate[0] < existing[0]:
                    best_by_candidate[candidate_idx] = candidate
        # Enforce one-to-one candidate feature / SfM landmark matches.
        ranked = sorted(
            ((distance, candidate_idx, landmark_id, xyz) for candidate_idx, (distance, landmark_id, xyz) in best_by_candidate.items()),
            key=lambda item: (item[0], item[1], item[2]),
        )
        used_landmarks: set[str] = set()
        points2d: list[tuple[float, float]] = []
        points3d: list[np.ndarray] = []
        landmark_ids: list[str] = []
        for _, candidate_idx, landmark_id, xyz in ranked:
            if landmark_id in used_landmarks:
                continue
            used_landmarks.add(landmark_id)
            x, y = keypoints[candidate_idx].pt
            points2d.append((float(x), float(y)))
            points3d.append(np.asarray(xyz, dtype=np.float64))
            landmark_ids.append(landmark_id)
        if points2d:
            p2 = np.asarray(points2d, dtype=np.float64)
            p3 = np.asarray(points3d, dtype=np.float64)
        else:
            p2 = np.empty((0, 2), dtype=np.float64)
            p3 = np.empty((0, 3), dtype=np.float64)
        pose = estimate_pose_from_correspondences(
            p2, p3,
            camera_model=camera_model,
            image_width=int(gray.shape[1]), image_height=int(gray.shape[0]),
            policy=self.policy,
        )
        if not pose.accepted:
            return DenseCandidateLocalizationResult(
                source_sequence_index=int(source_sequence_index), timestamp_ns=int(timestamp_ns),
                status=pose.rejection_status or DenseCandidateLocalizationStatus.REJECTED_POSE_ESTIMATION,
                accepted=False, is_global_anchor=False, global_workspace_order=None,
                keyframe_id=keyframe_id, T_reconstruction_from_camera=None,
                correspondence_count=pose.correspondence_count, inlier_count=pose.inlier_count,
                inlier_fraction=pose.inlier_fraction,
                median_reprojection_error_px=pose.median_reprojection_error_px,
                p90_reprojection_error_px=pose.p90_reprojection_error_px,
                anchor_workspace_orders_used=anchor_orders,
            )
        inlier_indices = np.flatnonzero(pose.inlier_mask)
        return DenseCandidateLocalizationResult(
            source_sequence_index=int(source_sequence_index), timestamp_ns=int(timestamp_ns),
            status=DenseCandidateLocalizationStatus.LOCALIZED,
            accepted=True, is_global_anchor=False, global_workspace_order=None,
            keyframe_id=keyframe_id, T_reconstruction_from_camera=pose.T_reconstruction_from_camera,
            correspondence_count=pose.correspondence_count, inlier_count=pose.inlier_count,
            inlier_fraction=pose.inlier_fraction,
            median_reprojection_error_px=pose.median_reprojection_error_px,
            p90_reprojection_error_px=pose.p90_reprojection_error_px,
            anchor_workspace_orders_used=anchor_orders,
            matched_landmark_ids=tuple(landmark_ids[i] for i in inlier_indices),
            matched_feature_xy=tuple(points2d[i] for i in inlier_indices),
        )


def global_anchor_localization_result(
    spec: DenseCandidateSpec,
    *,
    sfm_result: OfflineSfmResult,
) -> DenseCandidateLocalizationResult:
    if not spec.is_global_anchor:
        raise ReconstructionValidationError("global_anchor_localization_result requires an anchor spec.")
    camera = next(
        (item for item in sfm_result.cameras if item.keyframe_id == spec.global_keyframe_id),
        None,
    )
    if camera is None:
        raise ReconstructionValidationError("Global anchor has no authoritative SfM camera pose.")
    return DenseCandidateLocalizationResult(
        source_sequence_index=spec.source_sequence_index,
        timestamp_ns=spec.timestamp_ns,
        status=DenseCandidateLocalizationStatus.GLOBAL_ANCHOR_REUSED,
        accepted=True,
        is_global_anchor=True,
        global_workspace_order=spec.global_workspace_order,
        keyframe_id=str(camera.keyframe_id),
        T_reconstruction_from_camera=camera.T_reconstruction_from_camera,
        correspondence_count=0,
        inlier_count=0,
        inlier_fraction=1.0,
        median_reprojection_error_px=0.0,
        p90_reprojection_error_px=0.0,
        anchor_workspace_orders_used=(int(spec.global_workspace_order),),
    )


def summarize_localization_results(
    plan: DenseCandidatePlan,
    results: Sequence[DenseCandidateLocalizationResult],
    *,
    minimum_extra_localization_fraction: float = 0.80,
    minimum_accepted_extra_candidates: int = 10,
) -> dict[str, Any]:
    by_index = {int(item.source_sequence_index): item for item in results}
    missing = [item.source_sequence_index for item in plan.candidates if item.source_sequence_index not in by_index]
    if missing:
        raise ReconstructionValidationError(
            f"I4.1.6 localization summary is missing planned candidates: {missing[:8]}"
        )
    ordered = [by_index[item.source_sequence_index] for item in plan.candidates]
    extras = [item for item in ordered if not item.is_global_anchor]
    accepted_extras = [item for item in extras if item.accepted]
    fraction = float(len(accepted_extras) / max(1, len(extras))) if extras else 1.0
    medians = [float(item.median_reprojection_error_px) for item in accepted_extras if item.median_reprojection_error_px is not None]
    p90s = [float(item.p90_reprojection_error_px) for item in accepted_extras if item.p90_reprojection_error_px is not None]
    status_counts: dict[str, int] = {}
    for item in ordered:
        status_counts[item.status.value] = status_counts.get(item.status.value, 0) + 1
    full_span = plan.source_span_fraction >= 0.98
    localization_gate = (
        full_span
        and len(accepted_extras) >= int(minimum_accepted_extra_candidates)
        and fraction >= float(minimum_extra_localization_fraction)
    )
    return {
        "plan": plan.as_dict(),
        "planned_candidates": len(ordered),
        "global_anchor_candidates": sum(item.is_global_anchor for item in ordered),
        "extra_candidates": len(extras),
        "accepted_extra_candidates": len(accepted_extras),
        "extra_localization_fraction": fraction,
        "accepted_median_reprojection_error_p50_px": None if not medians else float(np.median(medians)),
        "accepted_p90_reprojection_error_p50_px": None if not p90s else float(np.median(p90s)),
        "status_counts": status_counts,
        "full_span": full_span,
        "localization_gate_pass": bool(localization_gate),
        "minimum_extra_localization_fraction": float(minimum_extra_localization_fraction),
        "minimum_accepted_extra_candidates": int(minimum_accepted_extra_candidates),
        "results": [item.as_dict(include_matches=False) for item in ordered],
    }
