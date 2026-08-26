from __future__ import annotations

from pathlib import Path


def test_i4_1_8a_runner_is_read_only_over_heavy_reconstruction_stages() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts/run_i4_1_8a_retained_core_target_diagnosis.py").read_text(encoding="utf-8")
    assert "run_f_confidence_artifact_filtering.py" not in runner
    assert "run_g_persistent_dense_rgb_geometry.py" not in runner
    assert "run_i2_full_video_quality_qualification.py" not in runner
    assert "sfm_rerun=False da3_rerun=False dense_pair_rerun=False fusion_rerun=False" in runner
    assert "production_target_changed=False orbit_thresholds_changed=False" in runner
    assert "metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False" in runner


def test_i4_1_8a_is_additive_and_does_not_edit_production_binding_policy() -> None:
    root = Path(__file__).resolve().parents[2]
    diagnosis = root / "core/reconstruction/offline_dense_i4_retained_core_target_diagnosis.py"
    production = root / "core/reconstruction/offline_dense_i4_production_binding.py"
    assert diagnosis.is_file()
    assert production.is_file()
    text = diagnosis.read_text(encoding="utf-8")
    assert "authorizes_target_change\": False" in text
    assert "production_target_changed\": False" in text
    assert "orbit_thresholds_changed\": False" in text


def test_i4_1_8a_preview_uses_reconstruction_canvas_and_multiple_diagnostic_markers() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts/run_i4_1_8a_target_preview.py").read_text(encoding="utf-8")
    assert "ReconstructionCanvas" in text
    assert "set_dense_rgb_points" in text
    assert "update_sparse_landmarks" in text
    assert "production_bound_trajectory_circle" in text
    assert "retained_core_camera_attention_candidate" in text
