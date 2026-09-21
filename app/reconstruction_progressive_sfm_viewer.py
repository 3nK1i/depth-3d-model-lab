"""I6.6D dependency-light progressive sparse-SfM viewer projection.

This module converts the frozen I6.6A provisional PyCOLMAP snapshot contract
into bounded NumPy buffers consumed by the existing ReconstructionCanvas.
It owns no Qt objects and grants no new authority: the resulting frame remains
provisional visualization only, arbitrary-scale, non-persistent, and distinct
from the admitted terminal sparse result.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.reconstruction.reconstruction_progressive_sfm_contract import (
    ProgressiveSfmSnapshot,
    ProgressiveSfmStage,
)

I66D_PHASE = "I6.6D"
I66D_VIEWER_PROFILE = "progressive_sparse_replace_v1"


def _readonly_contiguous(values: np.ndarray, *, dtype) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class ProgressiveSfmViewerFrame:
    """Bounded replace-style viewer buffers for one provisional SfM revision."""

    session_id: str
    revision: int
    stage: ProgressiveSfmStage
    stage_detail: str
    camera_matrices_reconstruction: np.ndarray
    camera_ids: tuple[str, ...]
    landmark_points_reconstruction: np.ndarray
    landmark_colors_rgb: np.ndarray | None
    registered_camera_total: int
    landmark_total: int
    observations_total: int
    mean_reprojection_error_px: float | None
    warnings: tuple[str, ...]
    provisional: bool = True
    viewer_profile: str = I66D_VIEWER_PROFILE

    @property
    def has_geometry(self) -> bool:
        return bool(
            self.camera_matrices_reconstruction.shape[0]
            or self.landmark_points_reconstruction.shape[0]
        )


def build_progressive_sfm_viewer_frame(
    snapshot: ProgressiveSfmSnapshot,
) -> ProgressiveSfmViewerFrame:
    """Project one validated I6.6A snapshot into bounded replacement buffers."""

    if not isinstance(snapshot, ProgressiveSfmSnapshot):
        raise TypeError("snapshot must be ProgressiveSfmSnapshot.")
    if not snapshot.provisional or snapshot.final_result_admitted:
        raise ValueError("I6.6D accepts provisional, non-admitted snapshots only.")

    geometry = snapshot.geometry
    if geometry is None:
        camera_matrices = np.zeros((0, 4, 4), dtype=np.float64)
        camera_ids: tuple[str, ...] = ()
        landmark_points = np.zeros((0, 3), dtype=np.float64)
        landmark_colors: np.ndarray | None = None
    else:
        camera_matrices = (
            np.asarray(
                [sample.T_reconstruction_from_camera for sample in geometry.camera_samples],
                dtype=np.float64,
            )
            if geometry.camera_samples
            else np.zeros((0, 4, 4), dtype=np.float64)
        )
        camera_ids = tuple(sample.source_image_name for sample in geometry.camera_samples)
        landmark_points = (
            np.asarray(
                [sample.position_reconstruction for sample in geometry.landmark_samples],
                dtype=np.float64,
            )
            if geometry.landmark_samples
            else np.zeros((0, 3), dtype=np.float64)
        )
        colors = tuple(sample.color_rgb for sample in geometry.landmark_samples)
        landmark_colors = None
        if colors and all(color is not None for color in colors):
            landmark_colors = np.asarray(colors, dtype=np.uint8)

    camera_matrices = _readonly_contiguous(camera_matrices, dtype=np.float64)
    landmark_points = _readonly_contiguous(landmark_points, dtype=np.float64)
    if landmark_colors is not None:
        landmark_colors = _readonly_contiguous(landmark_colors, dtype=np.uint8)

    if camera_matrices.shape != (len(camera_ids), 4, 4):
        raise ValueError("Progressive camera buffers are inconsistent.")
    if landmark_points.ndim != 2 or landmark_points.shape[1:] != (3,):
        raise ValueError("Progressive landmark buffers must be (N, 3).")
    if landmark_colors is not None and landmark_colors.shape != landmark_points.shape:
        raise ValueError("Progressive landmark colors must align with landmark points.")

    return ProgressiveSfmViewerFrame(
        session_id=snapshot.session_id,
        revision=snapshot.revision,
        stage=snapshot.stage,
        stage_detail=snapshot.stage_detail,
        camera_matrices_reconstruction=camera_matrices,
        camera_ids=camera_ids,
        landmark_points_reconstruction=landmark_points,
        landmark_colors_rgb=landmark_colors,
        registered_camera_total=snapshot.statistics.registered_cameras,
        landmark_total=snapshot.statistics.sparse_landmarks,
        observations_total=snapshot.statistics.observations,
        mean_reprojection_error_px=snapshot.statistics.mean_reprojection_error_px,
        warnings=snapshot.warnings,
    )


__all__ = [
    "I66D_PHASE",
    "I66D_VIEWER_PROFILE",
    "ProgressiveSfmViewerFrame",
    "build_progressive_sfm_viewer_frame",
]
