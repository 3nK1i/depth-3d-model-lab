from __future__ import annotations

import json

import pytest

from core.contracts import CalibrationStatus
from core.reconstruction import (
    LIVE_CALIBRATION_SCHEMA_VERSION,
    ReconstructionValidationError,
    live_camera_calibration_dict,
    load_live_camera_calibration,
)
from tests.reconstruction_live_helpers import make_calibrated_live_camera


def test_live_calibration_json_round_trip(tmp_path) -> None:
    camera = make_calibrated_live_camera()
    payload = live_camera_calibration_dict(camera)
    assert payload["schema_version"] == LIVE_CALIBRATION_SCHEMA_VERSION
    path = tmp_path / "camera.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_live_camera_calibration(path)
    assert loaded.camera_id == camera.camera_id
    assert loaded.image_size == camera.image_size
    assert loaded.intrinsics.calibration_status is CalibrationStatus.CALIBRATED
    assert loaded.intrinsics.calibration_record_id == camera.intrinsics.calibration_record_id


def test_live_calibration_rejects_missing_record_id(tmp_path) -> None:
    payload = live_camera_calibration_dict(make_calibrated_live_camera())
    payload["calibration_record_id"] = ""
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReconstructionValidationError):
        load_live_camera_calibration(path)
