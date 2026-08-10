from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_d3_consumes_existing_backend_quality_gate_without_reimplementing_thresholds() -> None:
    quality = (ROOT / "app/offline_reconstruction_quality.py").read_text(encoding="utf-8")
    page = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")

    assert 'sfm_summary.get("quality_assessment")' in quality
    assert "reject_below_registration_ratio" not in quality
    assert "degrade_above_mean_reprojection_error_px" not in quality
    assert "build_offline_sfm_health_presentation" in page
    assert "SPARSE SFM HEALTH" in page


def test_d3_is_presentation_only_and_preserves_non_metric_authority_language() -> None:
    quality = (ROOT / "app/offline_reconstruction_quality.py").read_text(encoding="utf-8").lower()
    page = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8").lower()

    assert "pycolmap" not in quality
    assert "not metrology" in page
    assert "not a metric-accuracy score" in quality
    assert "canonical_geometry_authoritative" not in quality
