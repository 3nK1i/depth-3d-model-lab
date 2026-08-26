from __future__ import annotations

from pathlib import Path


def _text(relative: str) -> str:
    return (Path(__file__).resolve().parents[2] / relative).read_text(encoding="utf-8")


def test_i42b_worker_uses_known_pinhole_k_exhaustive_matching_and_fixed_intrinsics() -> None:
    text = _text("scripts/reconstruction_pycolmap_temple_ring_sfm_worker.py")
    assert 'reader.camera_model = "PINHOLE"' in text
    assert "reader.camera_params" in text
    assert "pycolmap.match_exhaustive" in text
    assert "options.ba_refine_focal_length = False" in text
    assert "options.ba_refine_principal_point = False" in text
    assert "options.ba_refine_extra_params = False" in text
    assert "options.mapper.abs_pose_refine_focal_length = False" in text
    assert "options.mapper.abs_pose_refine_extra_params = False" in text


def test_i42b_worker_never_calls_dense_or_gt_pose_operations() -> None:
    text = _text("scripts/reconstruction_pycolmap_temple_ring_sfm_worker.py")
    forbidden_calls = (
        "patch_match_stereo(",
        "stereo_fusion(",
        "undistort_images(",
        "triangulate_points(",
        "bundle_adjustment(",
        "poisson_meshing(",
        "delaunay_meshing(",
    )
    for token in forbidden_calls:
        assert token not in text
    assert "ground_truth_validation_manifest" in text  # only in the fail-closed forbidden-token list
    assert '"ground_truth_extrinsics_consumed": False' in text
    assert '"metric_scale_claimed": False' in text
    assert '"surface_mesh_generated": False' in text


def test_i42b_orchestrator_requires_47_views_and_does_not_authorize_i43() -> None:
    text = _text("core/reconstruction/temple_ring_sfm.py")
    assert "TEMPLE_RING_EXPECTED_VIEWS = 47" in text
    assert '"permanent_global_view_cap": False' in text
    assert '"i4_3_provider_architecture_authorized": False' in text
    assert '"ground_truth_angles_consumed": False' in text
    assert "I4.2C held-out trajectory validation" in text
