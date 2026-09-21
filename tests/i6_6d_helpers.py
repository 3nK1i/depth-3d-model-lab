from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from core.reconstruction.reconstruction_progressive_sfm_contract import (
    I66A_PYCOLMAP_BACKEND_ID,
    ProgressiveSfmCameraSample,
    ProgressiveSfmGeometryPreview,
    ProgressiveSfmLandmarkSample,
    ProgressiveSfmSnapshot,
    ProgressiveSfmStage,
    ProgressiveSfmStatistics,
)

SESSION_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
REQUEST_SHA = "e" * 64


def make_progressive_view_snapshot(
    *,
    revision: int = 1,
    camera_sample_count: int = 3,
    landmark_sample_count: int = 5,
    registered_camera_total: int = 7,
    landmark_total: int = 101,
    stage: ProgressiveSfmStage = ProgressiveSfmStage.MAPPING,
    include_colors: bool = True,
) -> ProgressiveSfmSnapshot:
    cameras = []
    for index in range(camera_sample_count):
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 3] = float(index) * 0.2
        matrix[1, 3] = float(index) * 0.03
        cameras.append(
            ProgressiveSfmCameraSample(
                backend_image_id=index + 1,
                backend_camera_id=1,
                source_image_name=f"frame-{index:03d}.jpg",
                T_reconstruction_from_camera=matrix,
            )
        )
    landmarks = []
    for index in range(landmark_sample_count):
        color = (20 + index, 80 + index, 140 + index) if include_colors else None
        landmarks.append(
            ProgressiveSfmLandmarkSample(
                backend_point3d_id=1000 + index,
                position_reconstruction=np.asarray(
                    [float(index) * 0.1, float(index % 2) * 0.2, 1.0 + index * 0.01],
                    dtype=np.float64,
                ),
                color_rgb=color,
                track_length=3,
            )
        )
    geometry = ProgressiveSfmGeometryPreview(
        camera_samples=tuple(cameras),
        landmark_samples=tuple(landmarks),
    )
    timestamp = (
        datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
        + timedelta(milliseconds=revision)
    ).isoformat()
    return ProgressiveSfmSnapshot(
        session_id=SESSION_ID,
        revision=revision,
        timestamp_utc=timestamp,
        elapsed_ns=revision * 1_000_000,
        stage=stage,
        stage_detail=(
            "selected_model_before_final_result_admission"
            if stage is ProgressiveSfmStage.FINALIZING
            else "registered_image_snapshot"
        ),
        backend_id=I66A_PYCOLMAP_BACKEND_ID,
        backend_version="1",
        request_fingerprint_sha256=REQUEST_SHA,
        statistics=ProgressiveSfmStatistics(
            input_views=48,
            registered_cameras=max(registered_camera_total, camera_sample_count),
            sparse_landmarks=max(landmark_total, landmark_sample_count),
            observations=max(landmark_total, landmark_sample_count) * 3,
            mean_reprojection_error_px=0.45,
        ),
        progress_fraction=0.94 if stage is ProgressiveSfmStage.FINALIZING else 0.7,
        geometry=geometry,
    )
