from __future__ import annotations

from pathlib import Path


def test_i42a1_parser_recognizes_count_header_without_relaxing_47_view_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "core/reconstruction/temple_ring_preflight.py").read_text(encoding="utf-8")
    assert "count_header_present" in text
    assert "declared_view_count" in text
    assert "TEMPLE_RING_EXPECTED_VIEWS" in text
    assert "exact canonical templeR0001.png..templeR0047.png set" in text
    assert "calibration_count_header_present" in text
    assert "calibration_declared_view_count" in text


def test_i42a1_correction_does_not_authorize_reconstruction_or_provider_refactor() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "core/reconstruction/temple_ring_preflight.py").read_text(encoding="utf-8")
    for forbidden in (
        "incremental_mapping(",
        "bundle_adjustment(",
        "patch_match_stereo(",
        "stereo_fusion(",
    ):
        assert forbidden not in text
    assert '"sfm_started": False' in text
    assert '"mvs_started": False' in text
    assert '"i4_3_provider_architecture_authorized": False' in text
