from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.contracts import ImageSize
from core.reconstruction import (
    InputSourceKind,
    InputSourceMetadata,
    OfflineSfmResult,
    ReconstructionGauge,
    SfmCameraEstimate,
    SfmCameraModelEstimate,
    SfmLandmarkEstimate,
    SfmObservation,
    SfmStageStatistics,
    SfmWorkspaceImage,
    SfmWorkspaceManifest,
    TimestampOrigin,
)
from core.reconstruction.offline_dense_i4_decoupled_sampling_integration import (
    DenseCandidateLocalizationPolicy,
    DenseCandidateLocalizationStatus,
    DenseCandidatePlannerMode,
    build_dense_candidate_plan,
    estimate_pose_from_correspondences,
    global_anchor_localization_result,
    summarize_localization_results,
)

SESSION_ID = "00000000-0000-4000-8000-000000004116"


def _synthetic_sfm(tmp_path: Path, *, total_frames: int = 1517, fps: float = 29.97) -> OfflineSfmResult:
    width, height = 640, 480
    image_size = ImageSize(width, height)
    source = InputSourceMetadata(
        source_id="i4-1-6-video",
        kind=InputSourceKind.RECORDED_VIDEO,
        display_name="synthetic orbit",
        source_uri=str(tmp_path / "video.mp4"),
        expected_image_size=image_size,
        expected_frame_rate_hz=fps,
        timestamp_origin=TimestampOrigin.PRESENTATION_TIMESTAMP,
    )
    images_dir = tmp_path / "workspace/images"
    images_dir.mkdir(parents=True, exist_ok=True)
    indices = tuple(int(round(i * (total_frames - 1) / 39)) for i in range(40))
    workspace_images = []
    cameras = []
    for order, source_index in enumerate(indices):
        image_path = images_dir / f"kf_{order:02d}.png"
        cv2.imwrite(str(image_path), np.zeros((height, width, 3), dtype=np.uint8))
        keyframe_id = f"kf-{order:02d}"
        workspace_images.append(
            SfmWorkspaceImage(
                order=order,
                keyframe_id=keyframe_id,
                frame_id=f"frame-{order:02d}",
                source_sequence_index=source_index,
                timestamp_ns=int(round(source_index / fps * 1e9)),
                image_size=image_size,
                source_locator=str(tmp_path / "video.mp4"),
                staged_image_path=f"images/{image_path.name}",
                staged_sha256=(f"{order % 10}" * 64),
            )
        )
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = order * 0.1
        cameras.append(
            SfmCameraEstimate(
                estimate_id=f"pose-{order}",
                camera_id="camera-1",
                keyframe_id=keyframe_id,
                frame_id=f"frame-{order:02d}",
                backend_image_id=order + 1,
                T_reconstruction_from_camera=pose,
            )
        )
    workspace = SfmWorkspaceManifest(
        session_id=SESSION_ID,
        source=source,
        workspace_root=str(tmp_path / "workspace"),
        images_directory="images",
        database_path="database.db",
        sparse_directory="sparse",
        checkpoints_directory="checkpoints",
        reports_directory="reports",
        logs_directory="logs",
        images=tuple(workspace_images),
    )
    landmarks = []
    for index in range(60):
        observations = tuple(
            SfmObservation(
                observation_id=f"obs-{index}-{order}",
                landmark_id=f"lm-{index}",
                keyframe_id=f"kf-{order:02d}",
                feature_xy=(100.0 + index % 20, 100.0 + index // 20),
            )
            for order in (0, 1)
        )
        landmarks.append(
            SfmLandmarkEstimate(
                landmark_id=f"lm-{index}",
                track_id=f"track-{index}",
                position_reconstruction=np.array([index * 0.01, 0.1, 4.0 + index * 0.01]),
                observations=observations,
            )
        )
    return OfflineSfmResult(
        session_id=SESSION_ID,
        backend_id="pycolmap-offline-sfm",
        backend_version="test",
        gauge=ReconstructionGauge(),
        cameras=tuple(cameras),
        camera_models=(
            SfmCameraModelEstimate(
                camera_id="camera-1",
                backend_camera_id=1,
                backend_model_name="SIMPLE_RADIAL",
                image_size=image_size,
                parameters=(500.0, width / 2.0, height / 2.0, 0.0),
            ),
        ),
        landmarks=tuple(landmarks),
        statistics=SfmStageStatistics(
            input_keyframes=40,
            registered_keyframes=40,
            landmarks=len(landmarks),
            observations=sum(len(item.observations) for item in landmarks),
            elapsed_ns=1,
        ),
        workspace=workspace,
    )


def test_uniform_local_pool_preserves_full_span_and_local_density(tmp_path: Path) -> None:
    sfm = _synthetic_sfm(tmp_path)
    plan = build_dense_candidate_plan(
        sfm,
        source_total_frames=1517,
        source_fps=29.97,
        mode=DenseCandidatePlannerMode.UNIFORM_LOCALIZED_DENSE_POOL,
        dense_candidate_cap=120,
        local_target_fps=2.0,
    )
    payload = plan.as_dict()
    assert plan.candidate_count == 102
    assert plan.source_span_fraction == pytest.approx(1.0)
    assert payload["adjacent_gap_seconds"]["p50"] == pytest.approx(0.5005, abs=0.02)
    assert payload["adjacent_gap_seconds"]["fraction_le_0_75"] == pytest.approx(1.0)


def test_anchor_neighbor_plan_is_bounded_and_full_span(tmp_path: Path) -> None:
    sfm = _synthetic_sfm(tmp_path)
    plan = build_dense_candidate_plan(
        sfm,
        source_total_frames=1517,
        source_fps=29.97,
        mode=DenseCandidatePlannerMode.GLOBAL_ANCHORS_PLUS_LOCAL_NEIGHBORS,
        dense_candidate_cap=120,
        local_target_fps=2.0,
    )
    assert plan.candidate_count <= 120
    assert plan.source_span_fraction == pytest.approx(1.0)
    assert plan.extra_candidate_count > 0


def test_global_only_plan_reuses_sfm_anchor_identity(tmp_path: Path) -> None:
    sfm = _synthetic_sfm(tmp_path)
    plan = build_dense_candidate_plan(
        sfm,
        source_total_frames=1517,
        source_fps=29.97,
        mode=DenseCandidatePlannerMode.GLOBAL_ANCHORS_ONLY,
        dense_candidate_cap=120,
        local_target_fps=2.0,
    )
    assert plan.candidate_count == 40
    assert all(item.is_global_anchor for item in plan.candidates)
    result = global_anchor_localization_result(plan.candidates[0], sfm_result=sfm)
    assert result.status is DenseCandidateLocalizationStatus.GLOBAL_ANCHOR_REUSED
    assert result.accepted
    assert result.T_reconstruction_from_camera is not None


def test_pose_solver_recovers_synthetic_camera_pose(tmp_path: Path) -> None:
    sfm = _synthetic_sfm(tmp_path)
    model = sfm.camera_models[0]
    rng = np.random.default_rng(7)
    points3d = rng.uniform([-2.0, -1.0, 5.0], [2.0, 1.0, 10.0], size=(120, 3))
    rvec = np.array([[0.03], [-0.05], [0.02]], dtype=np.float64)
    tvec = np.array([[0.2], [-0.1], [0.4]], dtype=np.float64)
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    points2d, _ = cv2.projectPoints(points3d, rvec, tvec, K, np.zeros((5, 1)))
    points2d = points2d.reshape(-1, 2)
    points2d += rng.normal(0.0, 0.15, size=points2d.shape)
    policy = DenseCandidateLocalizationPolicy(
        minimum_correspondences=40,
        minimum_inliers=30,
        minimum_inlier_fraction=0.35,
    )
    evidence = estimate_pose_from_correspondences(
        points2d,
        points3d,
        camera_model=model,
        image_width=640,
        image_height=480,
        policy=policy,
    )
    assert evidence.accepted
    assert evidence.inlier_count >= 100
    assert evidence.median_reprojection_error_px is not None
    assert evidence.median_reprojection_error_px < 0.5
    assert evidence.T_reconstruction_from_camera is not None


def test_pose_solver_rejects_too_few_correspondences(tmp_path: Path) -> None:
    sfm = _synthetic_sfm(tmp_path)
    evidence = estimate_pose_from_correspondences(
        np.zeros((8, 2)),
        np.ones((8, 3)),
        camera_model=sfm.camera_models[0],
        image_width=640,
        image_height=480,
        policy=DenseCandidateLocalizationPolicy(),
    )
    assert not evidence.accepted
    assert evidence.rejection_status is DenseCandidateLocalizationStatus.REJECTED_INSUFFICIENT_MATCHES


def test_summary_requires_real_localization_fraction(tmp_path: Path) -> None:
    sfm = _synthetic_sfm(tmp_path)
    plan = build_dense_candidate_plan(
        sfm,
        source_total_frames=1517,
        source_fps=29.97,
        mode=DenseCandidatePlannerMode.GLOBAL_ANCHORS_ONLY,
        dense_candidate_cap=120,
        local_target_fps=2.0,
    )
    results = [global_anchor_localization_result(item, sfm_result=sfm) for item in plan.candidates]
    summary = summarize_localization_results(plan, results)
    assert summary["localization_gate_pass"] is False  # no *extra* localized evidence
    assert summary["full_span"] is True
