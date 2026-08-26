from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.contracts import ImageSize
from core.reconstruction import (
    InputSourceKind, InputSourceMetadata, OfflineSfmResult, ReconstructionGauge,
    SfmCameraEstimate, SfmCameraModelEstimate, SfmStageStatistics,
    SfmWorkspaceImage, SfmWorkspaceManifest, TimestampOrigin,
)
from core.reconstruction.offline_dense_i4_decoupled_sampling_integration import (
    DenseCandidatePlannerMode, build_dense_candidate_plan,
)

SESSION_ID = "00000000-0000-4000-8000-000000004117"


def _minimal_sfm(tmp_path: Path, total_frames: int) -> OfflineSfmResult:
    size = ImageSize(320, 240)
    source = InputSourceMetadata(
        source_id="stress", kind=InputSourceKind.RECORDED_VIDEO,
        display_name="stress", source_uri=str(tmp_path / "stress.mp4"),
        expected_image_size=size, expected_frame_rate_hz=30.0,
        timestamp_origin=TimestampOrigin.PRESENTATION_TIMESTAMP,
    )
    indices = tuple(int(round(i * (total_frames - 1) / 39)) for i in range(40))
    images = tuple(
        SfmWorkspaceImage(
            order=i, keyframe_id=f"kf-{i}", frame_id=f"f-{i}", source_sequence_index=idx,
            timestamp_ns=int(idx / 30.0 * 1e9), image_size=size,
            source_locator=str(tmp_path / "stress.mp4"), staged_image_path=f"images/{i}.png",
            staged_sha256=(str(i % 10) * 64),
        )
        for i, idx in enumerate(indices)
    )
    workspace = SfmWorkspaceManifest(
        session_id=SESSION_ID, source=source, workspace_root=str(tmp_path),
        images_directory="images", database_path="database.db", sparse_directory="sparse",
        checkpoints_directory="checkpoints", reports_directory="reports", logs_directory="logs",
        images=images,
    )
    cameras = tuple(
        SfmCameraEstimate(
            estimate_id=f"p-{i}", camera_id="c", keyframe_id=f"kf-{i}", frame_id=f"f-{i}",
            backend_image_id=i + 1, T_reconstruction_from_camera=np.eye(4),
        ) for i in range(40)
    )
    return OfflineSfmResult(
        session_id=SESSION_ID, backend_id="test", backend_version="test",
        gauge=ReconstructionGauge(), cameras=cameras,
        camera_models=(SfmCameraModelEstimate(camera_id="c", backend_camera_id=1, backend_model_name="SIMPLE_PINHOLE", image_size=size, parameters=(250.0,160.0,120.0)),),
        landmarks=(),
        statistics=SfmStageStatistics(input_keyframes=40, registered_keyframes=40, landmarks=0, observations=0, elapsed_ns=1),
        workspace=workspace,
    )


def test_planner_is_deterministic_across_repeated_long_video_cases(tmp_path: Path) -> None:
    for total in (1200, 1517, 2400, 3600):
        sfm = _minimal_sfm(tmp_path / str(total), total)
        a = build_dense_candidate_plan(sfm, source_total_frames=total, source_fps=30.0, mode=DenseCandidatePlannerMode.UNIFORM_LOCALIZED_DENSE_POOL, dense_candidate_cap=120, local_target_fps=2.0)
        b = build_dense_candidate_plan(sfm, source_total_frames=total, source_fps=30.0, mode=DenseCandidatePlannerMode.UNIFORM_LOCALIZED_DENSE_POOL, dense_candidate_cap=120, local_target_fps=2.0)
        assert [x.source_sequence_index for x in a.candidates] == [x.source_sequence_index for x in b.candidates]
        assert a.candidate_count <= 120
        assert a.source_span_fraction == 1.0


def test_neighbor_plan_never_exceeds_candidate_cap(tmp_path: Path) -> None:
    sfm = _minimal_sfm(tmp_path, 3600)
    for cap in (40, 60, 80, 120):
        plan = build_dense_candidate_plan(sfm, source_total_frames=3600, source_fps=30.0, mode=DenseCandidatePlannerMode.GLOBAL_ANCHORS_PLUS_LOCAL_NEIGHBORS, dense_candidate_cap=cap, local_target_fps=2.0)
        assert plan.candidate_count <= cap
        assert len({x.source_sequence_index for x in plan.candidates}) == plan.candidate_count
