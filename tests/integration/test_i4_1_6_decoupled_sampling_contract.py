from __future__ import annotations

from pathlib import Path


def test_i4_1_6_runner_is_additive_and_fail_closed() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/run_i4_1_6_decoupled_sampling_integration.py").read_text(encoding="utf-8")
    assert "single_authoritative_sfm_frame" in source
    assert '"production_sampling_integration_authorized": False' in source
    assert '"production_default_changed": False' in source
    assert '"dense_consistency_threshold_relaxation_authorized": False' in source
    assert '"pycolmap_rerun": False' in source
    assert '"da3_rerun": False' in source
    assert '"dense_pair_rerun": False' in source
    assert '"fusion_rerun": False' in source
    assert '"metric_scale_claimed": False' in source
    assert '"measurement_authoritative": False' in source
    assert '"surface_mesh_generated": False' in source


def test_i4_1_6_localizer_uses_existing_sfm_landmarks_not_second_map() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "core/reconstruction/offline_dense_i4_decoupled_sampling_integration.py").read_text(encoding="utf-8")
    assert "landmark.position_reconstruction" in source
    assert "solvePnPRansac" in source
    assert "invert_rigid_matrix" in source
    assert "incremental_mapping(" not in source
    assert "fuse_dense_frames(" not in source


def test_i4_1_6_default_compares_both_bounded_local_planners() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/run_i4_1_6_decoupled_sampling_integration.py").read_text(encoding="utf-8")
    assert 'default=["uniform_localized_dense_pool", "global_anchors_plus_local_neighbors"]' in source
    assert 'default=120' in source
    assert 'default=2.0' in source
