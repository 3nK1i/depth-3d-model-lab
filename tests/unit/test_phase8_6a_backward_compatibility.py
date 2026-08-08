from __future__ import annotations

import inspect
from pathlib import Path

from core.reconstruction import (
    CameraIntrinsicsPolicy,
    LiveSlamRequest,
    ReconstructionQualityProfile,
    live_camera_calibration_dict,
    make_usb_source_metadata,
)
from tests.reconstruction_helpers import SESSION_ID
from tests.reconstruction_live_helpers import make_calibrated_live_camera


def test_existing_calibrated_live_request_keeps_strict_default_policy() -> None:
    request = LiveSlamRequest(
        session_id=SESSION_ID,
        source=make_usb_source_metadata(0, width=320, height=240, frame_rate_hz=30.0),
        camera=make_calibrated_live_camera(),
        profile=ReconstructionQualityProfile.EFFICIENT,
        backend_id="development-live-slam-mock",
    )
    assert request.camera_intrinsics_policy is CameraIntrinsicsPolicy.CALIBRATED_ONLY
    assert request.camera.intrinsics.calibration_record_id == "calibration-live-test-1"


def test_existing_live_calibration_export_schema_is_unchanged() -> None:
    payload = live_camera_calibration_dict(make_calibrated_live_camera())
    assert payload["schema_version"] == 1
    assert payload["calibration_record_id"] == "calibration-live-test-1"
    assert "mode" not in payload
    assert "focal_length_relative_uncertainty" not in payload


def test_reconstruction_ui_remains_disabled() -> None:
    text = Path("app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    enabled_block = text.split("_ENABLED_MODES", 1)[1].split("}", 1)[0]
    assert "RECONSTRUCTION" not in enabled_block


def test_phase8_6a_does_not_import_native_stella_or_pycolmap() -> None:
    for path in Path("core/reconstruction").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import pycolmap" not in source
        assert "from pycolmap" not in source
        assert "import stella_vslam" not in source
        assert "from stella_vslam" not in source


def test_sourceframe_contract_is_not_modified_by_phase8_6a() -> None:
    from core.capture.frame import SourceFrame

    assert tuple(inspect.signature(SourceFrame).parameters) == ("image", "frame_id", "timestamp")
