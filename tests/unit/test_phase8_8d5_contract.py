from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_d5_is_presentation_only_and_keeps_semantic_authority_separate() -> None:
    semantic = _text("app/offline_reconstruction_semantics.py")
    page = _text("app/widgets/reconstruction_page.py")
    canvas = _text("app/widgets/viewport/vispy_canvas.py")

    assert "Semantic context only" in semantic
    assert "not inspection/QC decision" in semantic
    assert "cannot claim certified metrology" in semantic
    assert "SEMANTIC / OBJECTS" in page
    assert '"semantic_objects"' in page
    assert "set_semantic_geometry" in page
    assert "MAX_SEMANTIC_OBJECT_DISPLAY = 256" in canvas
    assert "not measurement boxes" in canvas


def test_d5_does_not_change_reconstruction_or_heavy_backend_boundaries() -> None:
    semantic = _text("app/offline_reconstruction_semantics.py")
    forbidden = (
        "import pycolmap",
        "from pycolmap",
        "import gsplat",
        "import nerfstudio",
        "import stella_vslam",
        "import depth_anything_3",
    )
    for token in forbidden:
        assert token not in semantic

    version = _text("core/version.py")
    assert '__version__ = "0.8.0"' in version
