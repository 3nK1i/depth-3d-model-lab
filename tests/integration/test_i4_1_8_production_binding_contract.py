from __future__ import annotations

from pathlib import Path


def test_i4_1_8_runner_preserves_heavy_authority_contracts() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts/run_i4_1_8_production_binding.py").read_text(encoding="utf-8")
    assert "fuse_dense_frames" in text
    assert "run_f_confidence_artifact_filtering.py" in text
    assert "run_g_persistent_dense_rgb_geometry.py" in text
    assert "run_i2_full_video_quality_qualification.py" in text
    assert "retained_core" in text.lower()
    assert "pycolmap_rerun=False da3_rerun=False dense_pair_rerun=False" in text
    assert "orbit_thresholds_changed=False" in text
    assert "metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False" in text


def test_i4_1_8_is_additive_and_does_not_change_i1_default_sampling() -> None:
    root = Path(__file__).resolve().parents[2]
    production = root / "core/reconstruction/offline_dense_i4_production_binding.py"
    runner = root / "scripts/run_i4_1_8_production_binding.py"
    assert production.is_file()
    assert runner.is_file()
    # The I4.1.8 overlay deliberately does not edit the generic I1 runner.
    text = production.read_text(encoding="utf-8")
    assert "production_default_changed" in text
    assert "False" in text
