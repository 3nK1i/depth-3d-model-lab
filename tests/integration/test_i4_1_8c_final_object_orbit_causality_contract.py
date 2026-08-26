from __future__ import annotations

from pathlib import Path


def test_i4_1_8c_runner_is_lightweight_and_never_reruns_production_or_searches_subsets() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts/run_i4_1_8c_final_object_orbit_causality.py").read_text(encoding="utf-8")
    assert "selected_camera_matrices" in text
    assert "production_selected_camera_matrices" in text
    assert "isolated_core_points_reconstruction" in text
    assert "run_f_confidence_artifact_filtering.py" not in text
    assert "run_g_persistent_dense_rgb_geometry.py" not in text
    assert "run_i2_full_video_quality_qualification.py" not in text
    assert "camera_subset_search_performed=False" in text
    assert "orbit_thresholds_changed=False" in text
    assert "sfm_rerun=False da3_rerun=False dense_pair_rerun=False fusion_rerun=False" in text


def test_i4_1_8c_policy_never_selects_target_or_camera_subset_using_orbit_gates() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "core/reconstruction/offline_dense_i4_final_object_orbit_causality.py").read_text(encoding="utf-8")
    assert '"orbit_metrics_used_to_select_object_support": False' in text
    assert '"orbit_metrics_used_to_select_target": False' in text
    assert '"camera_subset_search_performed": False' in text
    assert '"orbit_thresholds_changed": False' in text
    assert '"i4_2_detail_preservation_authorized": False' in text


def test_i4_1_8c_has_three_explicit_causality_outcomes() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "core/reconstruction/offline_dense_i4_final_object_orbit_causality.py").read_text(encoding="utf-8")
    assert "final_object_orbit_qualified" in text
    assert "production_binding_lost_object_angular_coverage" in text
    assert "qualified_path_object_orbit_insufficient" in text
