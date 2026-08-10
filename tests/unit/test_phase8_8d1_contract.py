from __future__ import annotations

from pathlib import Path


def test_d1_is_bounded_offline_visualization_only() -> None:
    root = Path(__file__).resolve().parents[2]
    worker = (root / "scripts/run_reconstruction_offline_job.py").read_text(encoding="utf-8")
    page = (root / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    canvas = (root / "app/widgets/viewport/vispy_canvas.py").read_text(encoding="utf-8")

    assert "OFFLINE_VISUALIZATION_LANDMARK_CAP = 8_192" in worker
    assert '"canonical_geometry_authoritative": False' in worker
    assert '"certified_metrology": False' in worker
    assert "offline_visualization_json" in worker
    assert "load_offline_sparse_visualization" in page
    assert "show_offline_sparse_reconstruction" in canvas
    loader = (root / "app/offline_reconstruction_visualization.py").read_text(encoding="utf-8").lower()
    assert "import pycolmap" not in loader
    assert "from pycolmap" not in loader
