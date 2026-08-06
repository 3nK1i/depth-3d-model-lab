from pathlib import Path
import numpy as np

from app.measurement_inspection_controller import InspectionControllerState
from app.widgets.inspection_page import InspectionPage
from core.inspection import InspectionBoxMethod, load_inspection_bundle
from tests.inspection_helpers import write_inspection_bundle


def test_inspection_page_progressive_workflow(qtbot, tmp_path: Path):
    page = InspectionPage()
    qtbot.addWidget(page)
    page.set_context(source_available=True, depth_model_id="depth-pro", depth_model_label="Depth Pro")
    assert page._capture_btn.isEnabled()
    assert not page._analyze_btn.isEnabled()

    page.set_captured_frame(np.full((48, 64, 3), 80, dtype=np.uint8))
    assert page._analyze_btn.isEnabled()

    summary, arrays = load_inspection_bundle(write_inspection_bundle(tmp_path / "bundle"))
    page.load_bundle(summary, arrays)
    assert page.selected_object is not None
    assert page._object_list.count() == 1
    assert page._measurement_table.rowCount() == 1
    assert "Not Calibrated" in page._uncertainty_notice.text()


def test_reset_mask_requests_authoritative_automatic_restore(qtbot, tmp_path: Path):
    page = InspectionPage()
    qtbot.addWidget(page)
    summary, arrays = load_inspection_bundle(write_inspection_bundle(tmp_path / "bundle", manual=True))
    page.load_bundle(summary, arrays)
    emitted = []
    page.recompute_requested.connect(emitted.append)
    page._reset_mask()
    assert page._reset_manual_requested
    assert page._recompute_btn.isEnabled()
    page._emit_recompute()
    assert emitted[-1]["reset_manual"] is True
    assert emitted[-1]["manual_mask"] is None


def test_operator_box_choice_enables_recompute(qtbot, tmp_path: Path):
    page = InspectionPage()
    qtbot.addWidget(page)
    summary, arrays = load_inspection_bundle(write_inspection_bundle(tmp_path / "bundle"))
    page.load_bundle(summary, arrays)
    index = page._box_method.findData(InspectionBoxMethod.CAMERA_ALIGNED)
    page._box_method.setCurrentIndex(index)
    assert page._recompute_btn.isEnabled()


def test_status_badge_and_progress_use_unambiguous_state_colours(qtbot):
    page = InspectionPage()
    qtbot.addWidget(page)

    page.set_progress(25, "perception", "Running perception")
    assert page._status_badge.property("statusKind") == "running"
    assert page._progress.property("statusKind") == "running"

    page.show_failure("Synthetic failure")
    assert page._status_badge.text() == "FAILED"
    assert page._status_badge.property("statusKind") == "error"
    assert page._progress.property("statusKind") == "error"

    page.set_controller_state(InspectionControllerState.COMPLETED)
    assert page._status_badge.text() == "COMPLETE"
    assert page._status_badge.property("statusKind") == "ready"
    assert page._progress.property("statusKind") == "ready"


def test_metric_distance_format_does_not_apply_millimetre_scale():
    formatted = InspectionPage._format_measurement({
        "kind": "camera_distance",
        "status": "available",
        "value_si": 2.9724206,
        "expanded_uncertainty_si": 0.5321164,
    })
    assert formatted == "2.972 ± 0.532 m"
