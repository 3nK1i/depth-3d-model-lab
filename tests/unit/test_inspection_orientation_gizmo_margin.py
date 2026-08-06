from pathlib import Path


def test_orientation_gizmo_has_safe_left_margin() -> None:
    source = Path("app/widgets/inspection_3d_canvas.py").read_text(encoding="utf-8")
    assert "self.setFixedSize(88, 72)" in source
    assert "origin = QPointF(34.0, 54.0)" in source
    assert "AlignLeft | Qt.AlignmentFlag.AlignBottom" in source
