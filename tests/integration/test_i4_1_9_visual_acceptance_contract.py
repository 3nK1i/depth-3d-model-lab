from __future__ import annotations

from pathlib import Path


def test_i419_runner_reuses_heavy_evidence_and_preserves_existing_kernels() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts/run_i4_1_9_visual_reconstruction_acceptance.py").read_text(encoding="utf-8")
    assert "run_reconstruction_pycolmap_smoke.py" not in text
    assert "run_i4_1_6_decoupled_sampling_integration.py" not in text
    assert "run_i4_1_6b_decoupled_dense_ab.py" not in text
    assert "fuse_dense_frames" in text
    assert "run_f_confidence_artifact_filtering.py" in text
    assert "run_g_persistent_dense_rgb_geometry.py" in text
    assert "run_i2_full_video_quality_qualification.py" in text
    assert "strict_pair_consistency_required_for_visual_path" in text
    assert "dense_consistency_thresholds_changed" in text
    assert "metric_scale_claimed" in text
    assert "measurement_authoritative" in text


def test_i419_closure_requires_manual_confirmation_and_keeps_geometry_separate() -> None:
    root = Path(__file__).resolve().parents[2]
    finalizer = (root / "scripts/run_i4_1_9_finalize_visual_acceptance.py").read_text(encoding="utf-8")
    assert "--confirm-manual-visual-pass" in finalizer
    assert "finalize_visual_acceptance" in finalizer
    core = (root / "core/reconstruction/offline_dense_i4_visual_acceptance.py").read_text(encoding="utf-8")
    assert '"i4_2_detail_preservation_authorized": True' in core
    assert '"geometry_accuracy_deferred": not strict_geometry_pass' in core
    assert '"metric_scale_claimed": False' in core
    assert '"certified_metrology": False' in core
    assert '"surface_mesh_generated": False' in core


def test_i419_does_not_modify_existing_strict_i417_or_i418_modules() -> None:
    root = Path(__file__).resolve().parents[2]
    # I4.1.9 is additive: it has its own core module/runners and consumes the
    # historical strict evidence instead of replacing those implementations.
    assert (root / "core/reconstruction/offline_dense_i4_visual_acceptance.py").is_file()
    assert (root / "scripts/run_i4_1_9_visual_reconstruction_acceptance.py").is_file()
    assert (root / "scripts/run_i4_1_7_clean_static_orbit_campaign.py").is_file()
    assert (root / "scripts/run_i4_1_8_production_binding.py").is_file()
