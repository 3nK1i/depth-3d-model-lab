from __future__ import annotations

import json
from pathlib import Path

from core.inspection import load_inspection_bundle
from tests.inspection_helpers import write_inspection_bundle


def test_legacy_inspection_bundle_remains_loadable_without_phase9_calibration_fields(tmp_path: Path) -> None:
    summary, _ = load_inspection_bundle(write_inspection_bundle(tmp_path / "legacy"))
    assert summary.calibration_profile_id == ""
    assert summary.calibration_record_id == ""
    assert summary.calibration_rectified is False


def test_phase9_calibration_provenance_round_trips_through_bundle_reader(tmp_path: Path) -> None:
    directory = write_inspection_bundle(tmp_path / "calibrated")
    path = directory / "inspection_bundle.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(
        {
            "intrinsics_source": "calibrated",
            "calibration_status": "calibrated",
            "calibration_profile_id": "phase9-profile",
            "calibration_record_id": "phase9-record",
            "calibration_method": "checkerboard",
            "calibration_validation_state": "validated",
            "calibration_application_mode": "same_aspect_full_frame_resize",
            "calibration_rectified": True,
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    summary, _ = load_inspection_bundle(directory)
    assert summary.intrinsics_source == "calibrated"
    assert summary.calibration_status == "calibrated"
    assert summary.calibration_profile_id == "phase9-profile"
    assert summary.calibration_record_id == "phase9-record"
    assert summary.calibration_method == "checkerboard"
    assert summary.calibration_validation_state == "validated"
    assert summary.calibration_application_mode == "same_aspect_full_frame_resize"
    assert summary.calibration_rectified is True
