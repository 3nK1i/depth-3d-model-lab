from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.reconstruction_canonical_cameras import (
    CANONICAL_CAMERA_POSE_CONVENTION,
    MAX_PERSISTED_CANONICAL_CAMERA_POSES,
    build_canonical_camera_trajectory_snapshot,
    load_persistent_canonical_camera_trajectory,
)
from app.reconstruction_visualization import keyframe_centers_in_viewer
from core.reconstruction import persistent_scene_from_pipeline_result
from core.reconstruction.orchestration import ReconstructionArtifactKind
from tests.reconstruction_f5_helpers import run_f5_fixture


def _matrices(count: int = 5) -> np.ndarray:
    poses = []
    for index in range(count):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = np.asarray(
            [-1.0 + index * 0.5, 0.12 * (index % 2), -0.35 + index * 0.18],
            dtype=np.float64,
        )
        poses.append(matrix)
    return np.asarray(poses, dtype=np.float64)


def _fixture(tmp_path: Path, *, include_matrices: bool = True):
    _fixture_data, _providers, result, snapshots = run_f5_fixture(
        tmp_path / "f5", request_id="phase9-3c-unit"
    )
    base = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    camera_artifact = next(
        item for item in base.artifacts if item.kind is ReconstructionArtifactKind.CAMERA_TRAJECTORY
    )
    if not include_matrices:
        return base, camera_artifact, snapshots, result
    matrices = _matrices()
    updated = dict(snapshots)
    updated["camera_trajectory"] = build_canonical_camera_trajectory_snapshot(
        matrices_reconstruction=matrices,
        camera_ids=tuple(f"frame-{index:03d}" for index in range(matrices.shape[0])),
        pose_layers=("optimized",) * matrices.shape[0],
        provider_id=camera_artifact.provider_id,
        reconstruction_frame_id="reconstruction-world",
    )
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=updated)
    return scene, camera_artifact, updated, result


def test_3c_snapshot_builder_preserves_pose_matrices_without_authority_promotion() -> None:
    matrices = _matrices(3)
    snapshot = build_canonical_camera_trajectory_snapshot(
        matrices_reconstruction=matrices,
        camera_ids=("a", "b", "c"),
        pose_layers=("raw", "optimized", "optimized"),
        provider_id="pose-provider",
        reconstruction_frame_id="reconstruction-world",
    )

    assert snapshot["pose_convention"] == CANONICAL_CAMERA_POSE_CONVENTION
    assert snapshot["frame_ids"] == ["a", "b", "c"]
    np.testing.assert_allclose(np.asarray(snapshot["T_reconstruction_from_camera"]), matrices)
    assert snapshot["canonical_geometry_authoritative"] is False
    assert snapshot["certified_metrology"] is False
    assert snapshot["presentation_role"] == "camera_pose_context"


def test_3c_persistent_camera_loader_returns_validated_scene_bound_presentation(tmp_path: Path) -> None:
    scene, artifact, _snapshots, _result = _fixture(tmp_path)
    presentation = load_persistent_canonical_camera_trajectory(scene)

    assert presentation is not None
    assert presentation.scene_id == scene.scene_id
    assert presentation.artifact_id == artifact.artifact_id
    assert presentation.provider_id == artifact.provider_id
    assert presentation.pose_convention == CANONICAL_CAMERA_POSE_CONVENTION
    assert presentation.reconstruction_frame_id == "reconstruction-world"
    assert presentation.persisted_pose_count == 5
    assert presentation.total_pose_count == 5
    assert presentation.trajectory_available is True
    assert presentation.certified_metrology is False
    np.testing.assert_allclose(
        presentation.positions_viewer,
        keyframe_centers_in_viewer(presentation.matrices_reconstruction),
    )


def test_3c_legacy_camera_artifact_without_pose_matrices_remains_valid_but_not_renderable(tmp_path: Path) -> None:
    scene, _artifact, _snapshots, _result = _fixture(tmp_path, include_matrices=False)

    assert scene.canonical_geometry_authoritative is True
    assert load_persistent_canonical_camera_trajectory(scene) is None
    assert scene.certified_metrology is False


def test_3c_provenance_only_camera_snapshot_does_not_invent_matrices(tmp_path: Path) -> None:
    scene, artifact, snapshots, result = _fixture(tmp_path, include_matrices=False)
    updated = dict(snapshots)
    updated["camera_trajectory"] = {
        "pose_convention": CANONICAL_CAMERA_POSE_CONVENTION,
        "frame_ids": ["frame-a", "frame-b"],
        "frame_count": 2,
        "pose_layers": ["optimized", "optimized"],
        "source_provider_id": artifact.provider_id,
        "certified_metrology": False,
    }
    provenance_only = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=updated)

    assert load_persistent_canonical_camera_trajectory(provenance_only) is None


def test_3c_matrix_snapshot_requires_exact_pose_convention(tmp_path: Path) -> None:
    _scene, _artifact, snapshots, result = _fixture(tmp_path)
    updated = dict(snapshots)
    bad = dict(updated["camera_trajectory"])
    bad["pose_convention"] = "T_camera_from_reconstruction"
    updated["camera_trajectory"] = bad
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=updated)

    with pytest.raises(ValueError, match="pose_convention"):
        load_persistent_canonical_camera_trajectory(scene)


def test_3c_matrix_snapshot_rejects_provider_provenance_mismatch(tmp_path: Path) -> None:
    _scene, _artifact, snapshots, result = _fixture(tmp_path)
    updated = dict(snapshots)
    bad = dict(updated["camera_trajectory"])
    bad["source_provider_id"] = "wrong-provider"
    updated["camera_trajectory"] = bad
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=updated)

    with pytest.raises(ValueError, match="provider provenance"):
        load_persistent_canonical_camera_trajectory(scene)


def test_3c_matrix_snapshot_rejects_nonrigid_pose(tmp_path: Path) -> None:
    _scene, _artifact, snapshots, result = _fixture(tmp_path)
    updated = dict(snapshots)
    bad = dict(updated["camera_trajectory"])
    matrices = np.asarray(bad["T_reconstruction_from_camera"], dtype=np.float64)
    matrices[0, 0, 0] = 2.0
    bad["T_reconstruction_from_camera"] = matrices.tolist()
    updated["camera_trajectory"] = bad
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=updated)

    with pytest.raises(ValueError, match="orthonormal"):
        load_persistent_canonical_camera_trajectory(scene)


def test_3c_snapshot_builder_enforces_bounded_persistent_pose_count() -> None:
    matrices = np.repeat(
        np.eye(4, dtype=np.float64)[None, :, :],
        MAX_PERSISTED_CANONICAL_CAMERA_POSES + 1,
        axis=0,
    )
    ids = tuple(f"frame-{index}" for index in range(matrices.shape[0]))
    with pytest.raises(ValueError, match="safety bound"):
        build_canonical_camera_trajectory_snapshot(
            matrices_reconstruction=matrices,
            camera_ids=ids,
            provider_id="provider",
        )


def test_3c_camera_artifact_cannot_be_promoted_to_canonical_authority(tmp_path: Path) -> None:
    scene, artifact, _snapshots, _result = _fixture(tmp_path)
    assert artifact.canonical_geometry_authoritative is False
    assert scene.canonical_geometry_authoritative is True
    presentation = load_persistent_canonical_camera_trajectory(scene)
    assert presentation is not None and presentation.certified_metrology is False

@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("canonical_geometry_authoritative", "canonical geometry authority"),
        ("certified_metrology", "certified metrology"),
    ),
)
def test_3c_matrix_snapshot_rejects_camera_authority_promotion(
    tmp_path: Path, field: str, message: str
) -> None:
    _scene, _artifact, snapshots, result = _fixture(tmp_path)
    updated = dict(snapshots)
    bad = dict(updated["camera_trajectory"])
    bad[field] = True
    updated["camera_trajectory"] = bad
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=updated)

    with pytest.raises(ValueError, match=message):
        load_persistent_canonical_camera_trajectory(scene)
