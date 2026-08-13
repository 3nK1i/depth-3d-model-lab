from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from core.contracts import ImageSize
from core.reconstruction import (
    CalibrationProfileFileFormat,
    CameraCalibrationMethod,
    CameraCalibrationProfileStore,
    CameraCalibrationValidationState,
    CameraDistortionModel,
    ReconstructionValidationError,
    create_manual_camera_calibration_profile,
    import_camera_calibration_file,
    live_camera_calibration_dict,
    make_usb_source_metadata,
)
from tests.reconstruction_live_helpers import make_calibrated_live_camera


CREATED_AT = "2026-08-13T16:30:00+00:00"


def _manual_profile(*, profile_id: str = "manual-profile"):
    return create_manual_camera_calibration_profile(
        width=640,
        height=480,
        fx=520.0,
        fy=522.0,
        cx=320.0,
        cy=240.0,
        distortion_model=CameraDistortionModel.BROWN_CONRADY,
        distortion_coefficients=(-0.01, 0.002, 0.0, 0.0, 0.0),
        profile_id=profile_id,
        calibration_record_id=f"{profile_id}-record",
        camera_id="usb-camera-0",
        provenance="pytest-manual-entry",
        created_at_utc=CREATED_AT,
    )


def test_manual_entry_creates_calibrated_intrinsics_but_unvalidated_profile() -> None:
    profile = _manual_profile()

    assert profile.method is CameraCalibrationMethod.MANUAL
    assert profile.validation_state is CameraCalibrationValidationState.UNVALIDATED
    assert profile.is_accepted is False
    assert profile.has_phase9_validation_evidence is False
    assert profile.camera.image_size == ImageSize(640, 480)
    assert profile.camera.intrinsics.calibration_record_id == "manual-profile-record"
    assert profile.quality is None


def test_manual_entry_binds_source_without_storing_source_uri() -> None:
    source = make_usb_source_metadata(0, width=640, height=480, frame_rate_hz=30.0)
    profile = create_manual_camera_calibration_profile(
        width=640,
        height=480,
        fx=520.0,
        fy=522.0,
        cx=320.0,
        cy=240.0,
        profile_id="source-bound",
        calibration_record_id="source-bound-record",
        source=source,
        provenance="pytest",
        created_at_utc=CREATED_AT,
    )

    assert profile.source_binding is not None
    assert profile.source_binding.source_id == source.source_id
    assert profile.source_binding.fingerprint == source.fingerprint
    assert "source_uri" not in json.dumps(profile.as_dict())


def test_manual_entry_rejects_source_resolution_mismatch() -> None:
    source = make_usb_source_metadata(0, width=320, height=240, frame_rate_hz=30.0)
    with pytest.raises(ReconstructionValidationError, match="resolution"):
        create_manual_camera_calibration_profile(
            width=640,
            height=480,
            fx=520.0,
            fy=522.0,
            cx=320.0,
            cy=240.0,
            profile_id="mismatch",
            calibration_record_id="mismatch-record",
            source=source,
            provenance="pytest",
            created_at_utc=CREATED_AT,
        )


def test_manual_entry_rejects_invalid_optics_and_backend_specific_distortion() -> None:
    with pytest.raises(Exception):
        create_manual_camera_calibration_profile(
            width=640,
            height=480,
            fx=-1.0,
            fy=522.0,
            cx=320.0,
            cy=240.0,
            profile_id="bad-focal",
            calibration_record_id="bad-focal-record",
            provenance="pytest",
            created_at_utc=CREATED_AT,
        )

    with pytest.raises(ReconstructionValidationError, match="portable distortion model"):
        create_manual_camera_calibration_profile(
            width=640,
            height=480,
            fx=520.0,
            fy=522.0,
            cx=320.0,
            cy=240.0,
            distortion_model=CameraDistortionModel.BACKEND_SPECIFIC,
            distortion_coefficients=(1.0,),
            profile_id="backend-specific",
            calibration_record_id="backend-specific-record",
            provenance="pytest",
            created_at_utc=CREATED_AT,
        )


def test_native_phase9_import_preserves_original_evidence(tmp_path: Path) -> None:
    profile = replace(
        _manual_profile(profile_id="native-import"),
        validation_state=CameraCalibrationValidationState.VALIDATED,
    )
    path = tmp_path / "native.json"
    path.write_text(json.dumps(profile.as_dict()), encoding="utf-8")

    imported = import_camera_calibration_file(path)

    assert imported.file_format is CalibrationProfileFileFormat.PHASE9_PROFILE
    assert imported.profile.as_dict() == profile.as_dict()
    assert imported.profile.method is CameraCalibrationMethod.MANUAL
    assert imported.profile.validation_state is CameraCalibrationValidationState.VALIDATED


def test_legacy_phase8_import_remains_legacy_compatible(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(live_camera_calibration_dict(make_calibrated_live_camera())),
        encoding="utf-8",
    )

    imported = import_camera_calibration_file(
        path,
        legacy_profile_id="legacy-import",
        created_at_utc=CREATED_AT,
    )

    assert imported.file_format is CalibrationProfileFileFormat.PHASE8_LIVE_JSON
    assert imported.profile.profile_id == "legacy-import"
    assert imported.profile.method is CameraCalibrationMethod.LEGACY_LIVE_JSON
    assert imported.profile.validation_state is CameraCalibrationValidationState.LEGACY_COMPATIBLE
    assert imported.profile.has_phase9_validation_evidence is False


def test_unknown_import_format_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unknown.json"
    path.write_text('{"schema_version": 999, "foo": "bar"}', encoding="utf-8")

    with pytest.raises(ReconstructionValidationError, match="Unsupported camera calibration file"):
        import_camera_calibration_file(path)


def test_profile_store_round_trip_list_and_duplicate_protection(tmp_path: Path) -> None:
    store = CameraCalibrationProfileStore(tmp_path / "profiles")
    first = _manual_profile(profile_id="first")
    second = _manual_profile(profile_id="second")
    second = replace(second, created_at_utc="2026-08-13T16:31:00+00:00")

    first_path = store.save(first)
    second_path = store.save(second)

    assert first_path.parent == store.root
    assert second_path.parent == store.root
    assert first_path.name.endswith(".json")
    assert "first" not in first_path.name
    assert store.contains("first") is True
    assert store.load("first").as_dict() == first.as_dict()
    assert [item.profile_id for item in store.list_profiles()] == ["first", "second"]

    with pytest.raises(ReconstructionValidationError, match="already exists"):
        store.save(first)


def test_profile_store_key_cannot_escape_root(tmp_path: Path) -> None:
    store = CameraCalibrationProfileStore(tmp_path / "profiles")
    profile = _manual_profile(profile_id="../../outside")
    path = store.save(profile)

    assert path.parent == store.root
    assert path.resolve().is_relative_to(store.root)
    assert not (tmp_path / "outside").exists()
    assert store.load("../../outside").profile_id == "../../outside"


def test_profile_store_overwrite_is_explicit(tmp_path: Path) -> None:
    store = CameraCalibrationProfileStore(tmp_path / "profiles")
    profile = _manual_profile(profile_id="replace-me")
    store.save(profile)
    updated = replace(profile, notes="updated")

    path = store.save(updated, overwrite=True)
    assert store.load("replace-me").notes == "updated"
    assert not tuple(path.parent.glob("*.tmp"))


def test_store_import_persists_detected_profile(tmp_path: Path) -> None:
    store = CameraCalibrationProfileStore(tmp_path / "profiles")
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(
        json.dumps(live_camera_calibration_dict(make_calibrated_live_camera())),
        encoding="utf-8",
    )

    imported = store.import_file(
        legacy_path,
        legacy_profile_id="stored-legacy",
        created_at_utc=CREATED_AT,
    )

    assert imported.profile.profile_id == "stored-legacy"
    assert store.load("stored-legacy").validation_state is CameraCalibrationValidationState.LEGACY_COMPATIBLE
