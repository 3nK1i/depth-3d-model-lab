"""I6.7D bounded, Qt-free projection of provisional dense transport snapshots.

No camera, metric, persistent, final-admission, or measurement authority is gained.
The GPU/VisPy upload is owned by the main-thread viewport, never by this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.reconstruction.reconstruction_progressive_dense_contract import (
    ProgressiveDenseSnapshot,
    ProgressiveDenseStage,
)
from core.reconstruction.reconstruction_progressive_dense_transport import (
    I67C_MAX_GEOMETRY_PREVIEW_POINTS,
)

I67D_PHASE = "I6.7D_INCREMENTAL_DENSE_VIEWER_INTEGRATION"
I67D_MAX_DISPLAY_POINTS = I67C_MAX_GEOMETRY_PREVIEW_POINTS
I67D_VIEWER_PROFILE = "provisional_dense_replace_gpu_v1"


@dataclass(frozen=True, slots=True)
class ProgressiveDenseViewerFrame:
    dense_run_id: str
    source_sfm_session_id: str
    source_sparse_result_sha256: str
    reconstruction_frame_id: str
    revision: int
    stage: ProgressiveDenseStage
    stage_detail: str
    current_dense_points: int
    points_viewer: np.ndarray | None
    colors_rgba: np.ndarray | None
    provisional: bool = True
    viewer_profile: str = I67D_VIEWER_PROFILE

    @property
    def has_geometry_update(self) -> bool:
        return self.points_viewer is not None

    @property
    def display_point_count(self) -> int:
        return 0 if self.points_viewer is None else int(self.points_viewer.shape[0])


def build_progressive_dense_viewer_frame(
    snapshot: ProgressiveDenseSnapshot,
) -> ProgressiveDenseViewerFrame:
    """Project at most 4096 points directly to contiguous float32 GPU buffers.

    Status-only revisions contain no arrays: their presentation must not force a
    geometry rebuild or GPU buffer upload. This is a bounded visual preview, not
    an application-side copy of the full dense reconstruction.
    """
    if not isinstance(snapshot, ProgressiveDenseSnapshot):
        raise TypeError("snapshot must be ProgressiveDenseSnapshot.")
    if (not snapshot.provisional or snapshot.final_dense_result_admitted
            or snapshot.persistent_dense_scene or snapshot.metric_scale_claimed
            or snapshot.measurement_authoritative or snapshot.certified_metrology):
        raise ValueError("I6.7D requires provisional, non-authoritative dense snapshots.")
    preview = snapshot.geometry
    points: np.ndarray | None = None
    colors: np.ndarray | None = None
    if preview is not None:
        samples = preview.point_samples
        if len(samples) > I67D_MAX_DISPLAY_POINTS:
            raise ValueError("Dense preview exceeds frozen I6.7C point budget.")
        # Vectorize CPU staging: one bulk conversion, one axis transform and
        # one color conversion. Per-sample array creation was ~7x slower on the
        # local 4096-point projection microbenchmark; no GPU transfer occurs here.
        reconstruction = np.asarray(
            [sample.position_reconstruction for sample in samples], dtype=np.float32
        ).reshape((-1, 3))
        points = np.empty((len(samples), 3), dtype=np.float32)
        points[:, 0] = reconstruction[:, 0]
        points[:, 1] = reconstruction[:, 2]
        points[:, 2] = -reconstruction[:, 1]
        if not np.isfinite(points).all():
            raise ValueError("Dense preview coordinates exceed finite float32 GPU range.")
        colors = np.empty((len(samples), 4), dtype=np.float32)
        rgb = np.asarray(
            [sample.color_rgb if sample.color_rgb is not None else (143, 199, 240)
             for sample in samples],
            dtype=np.float32,
        ).reshape((-1, 3))
        np.multiply(rgb, np.float32(1.0 / 255.0), out=colors[:, :3])
        colors[:, 3] = 0.88  # constant visualization opacity; not confidence
        points.setflags(write=False)
        colors.setflags(write=False)
    return ProgressiveDenseViewerFrame(
        dense_run_id=snapshot.dense_run_id,
        source_sfm_session_id=snapshot.source_sfm_session_id,
        source_sparse_result_sha256=snapshot.source_sparse_result_sha256,
        reconstruction_frame_id=snapshot.reconstruction_frame_id,
        revision=snapshot.revision,
        stage=snapshot.stage,
        stage_detail=snapshot.stage_detail,
        current_dense_points=snapshot.statistics.current_dense_points,
        points_viewer=points,
        colors_rgba=colors,
    )


__all__ = [
    "I67D_PHASE", "I67D_MAX_DISPLAY_POINTS", "I67D_VIEWER_PROFILE",
    "ProgressiveDenseViewerFrame", "build_progressive_dense_viewer_frame",
]
