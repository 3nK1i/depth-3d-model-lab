from pathlib import Path

from PyQt6.QtGui import QPixmap

from app.widgets.inspection_canvas import InspectionImageCanvas
from core.inspection import load_inspection_bundle
from tests.inspection_helpers import write_inspection_bundle


def test_label_overlay_renders_with_fractional_scaled_coordinates(qtbot, tmp_path: Path):
    summary, arrays = load_inspection_bundle(write_inspection_bundle(tmp_path / "bundle"))
    canvas = InspectionImageCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(733, 487)  # Deliberately creates a fractional image scale.
    canvas.set_bundle(arrays["aligned_rgb"], summary.objects, arrays)
    canvas.set_overlay_options(masks=True, boxes=True, labels=True, uncertainty=True)

    target = QPixmap(canvas.size())
    target.fill()
    canvas.render(target)

    assert not target.isNull()
    assert canvas._image_rect.width() > 0.0
