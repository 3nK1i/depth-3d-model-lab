from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_d4_is_presentation_only_and_does_not_add_backend_or_metric_authority() -> None:
    spatial = (ROOT / "app/offline_reconstruction_spatial.py").read_text(encoding="utf-8").lower()
    page = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8").lower()

    assert "pycolmap" not in spatial
    assert "core.reconstruction" not in spatial
    assert "presentation envelope only" in spatial
    assert "not object dimensions" in spatial
    assert "not object dimensions" in page
    assert "metres" in spatial
    assert "canonical_geometry_authoritative" not in spatial
    assert "certified_metrology" not in spatial


def test_d4_canvas_contract_keeps_sparse_envelope_separate_from_authoritative_geometry() -> None:
    canvas = (ROOT / "app/widgets/viewport/vispy_canvas.py").read_text(encoding="utf-8")
    page = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")

    assert "set_offline_sparse_envelope" in canvas
    assert "clear_offline_sparse_envelope" in canvas
    assert "offline_sparse_envelope_available" in canvas
    assert "SPARSE SPATIAL COVERAGE" in page
    assert 'self._viewport_tool_buttons["offline_envelope"]' in page
