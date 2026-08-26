from __future__ import annotations

from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_i42e2a_runner_treats_full_view_coverage_as_diagnostic_not_gate() -> None:
    source = (_root() / "scripts/run_i4_2e_temple_ring_detail_qualification.py").read_text(encoding="utf-8")
    assert '"native_visibility_full_view_coverage_required": False' in source
    assert "native_visibility_observed_image_indices" in source
    assert "did not cover the exact 47-view MVS index set" not in source
    assert "did not retain all 47 MVS image indices" not in source


def test_i42e2a_worker_reports_observed_coverage_without_promoting_it_to_validity() -> None:
    source = (_root() / "scripts/reconstruction_pycolmap_temple_ring_detail_support_worker.py").read_text(encoding="utf-8")
    assert '"native_visibility_observed_image_indices": observed_image_indices.tolist()' in source
    assert '"native_visibility_indices_within_expected_range": True' in source
    assert '"native_visibility_full_view_coverage_required": False' in source
    assert "does not cover the exact expected MVS image-index set" not in source
