from __future__ import annotations

from pathlib import Path


def test_i42a_preflight_is_read_only_and_does_not_start_reconstruction() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "core/reconstruction/temple_ring_preflight.py").read_text(encoding="utf-8")
    for forbidden in (
        "incremental_mapping(",
        "bundle_adjustment(",
        "undistort_images(",
        "patch_match_stereo(",
        "stereo_fusion(",
        "fuse_dense_frames(",
    ):
        assert forbidden not in text
    assert '"sfm_started": False' in text
    assert '"mvs_started": False' in text
    assert '"production_filter_policy_changed": False' in text
    assert '"i4_3_provider_architecture_authorized": False' in text


def test_i42a_reconstruction_manifest_explicitly_quarantines_gt_extrinsics() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "core/reconstruction/temple_ring_preflight.py").read_text(encoding="utf-8")
    assert '"ground_truth_extrinsics_included": False' in text
    assert '"ground_truth_camera_centers_included": False' in text
    assert '"ground_truth_angles_included": False' in text
    assert '"sfm_pose_input_authorized": False' in text
    assert '"bundle_adjustment_pose_prior_authorized": False' in text


def test_i42a_runner_has_unit_integration_stress_modes() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts/run_i4_2a_tests.sh").read_text(encoding="utf-8")
    assert "test_i4_2a_temple_ring_preflight.py" in text
    assert "test_i4_2a_temple_ring_preflight_contract.py" in text
    assert "test_i4_2a_temple_ring_preflight_stress.py" in text
    assert "focused|stress|all" in text
