from __future__ import annotations

from pathlib import Path


def _text(relative: str) -> str:
    return (Path(__file__).resolve().parents[2] / relative).read_text(encoding="utf-8")


def test_i42c_pose_export_worker_is_read_only_and_has_no_reconstruction_or_dense_calls() -> None:
    text = _text("scripts/reconstruction_pycolmap_temple_ring_pose_export_worker.py")
    assert "pycolmap.Reconstruction(sparse)" in text
    for token in (
        "incremental_mapping(",
        "bundle_adjustment(",
        "triangulate_points(",
        "undistort_images(",
        "patch_match_stereo(",
        "stereo_fusion(",
        "poisson_meshing(",
        "delaunay_meshing(",
    ):
        assert token not in text
    assert '"authoritative_sparse_modified": False' in text
    assert "R_camera_from_world" not in text
    assert "camera_center_world" not in text


def test_i42c_alignment_is_center_only_proper_sim3_and_never_rewrites_authoritative_model() -> None:
    text = _text("core/reconstruction/temple_ring_trajectory_validation.py")
    assert "Umeyama least-squares Sim(3) on all 47 corresponding camera centres" in text
    assert '"ground_truth_rotations_used_for_alignment": False' in text
    assert '"applied_to_authoritative_sparse_model": False' in text
    assert '"similarity_applied_to_authoritative_sparse_model": False' in text
    assert '"authoritative_camera_poses_rewritten": False' in text
    assert '"benchmark_similarity_scale_is_metric_authority": False' in text


def test_i42c_does_not_authorize_mvs_filter_changes_or_i43_until_trajectory_passes() -> None:
    text = _text("core/reconstruction/temple_ring_trajectory_validation.py")
    assert '"mvs_started": False' in text
    assert '"production_filter_policy_changed": False' in text
    assert '"dense_provider_promoted": False' in text
    assert '"i4_3_provider_architecture_authorized": False' in text
    assert "I4.2D TempleRing MVS dense reconstruction" in text
