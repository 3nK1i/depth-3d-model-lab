from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.reconstruction_one_start_preflight import (
    CANONICAL_CAMERA_AUTHORITY_OWNER,
    ONE_START_EXECUTION_SCOPE,
    PRODUCTION_AUTO_POLICY_SCOPE,
    OneStartBlockerCode,
    OneStartPreflightStatus,
    build_offline_one_start_preflight,
    offline_one_start_transition_allowed,
)
from core.reconstruction import ReconstructionQualityProfile

SESSION = "11111111-1111-4111-8111-111111111111"


def _selection(path: Path, *, workflow: str = "image_sequence", ready: bool = True, item_count: int = 4):
    return SimpleNamespace(
        workflow=SimpleNamespace(value=workflow),
        ready=ready,
        display_name=path.name,
        detail=f"{item_count} source items",
        path=path,
        item_count=item_count,
    )


def _images(tmp_path: Path, *, count: int = 4) -> Path:
    root = tmp_path / "images"
    root.mkdir()
    for index in range(count):
        (root / f"frame-{index:03d}.jpg").write_bytes(b"fixture")
    return root


def _sfm_python(tmp_path: Path) -> Path:
    path = tmp_path / "pycolmap-python"
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _build(tmp_path: Path, **overrides):
    images = overrides.pop("images", None)
    if images is None:
        images = _images(tmp_path)
    sfm_python = overrides.pop("sfm_python", _sfm_python(tmp_path))
    output_root = overrides.pop("output_root", tmp_path / "outputs")
    selection = overrides.pop("selection", _selection(images))
    return build_offline_one_start_preflight(
        source_selection=selection,
        controller_shutdown=overrides.pop("controller_shutdown", False),
        live_active=overrides.pop("live_active", False),
        offline_active=overrides.pop("offline_active", False),
        da3_active=overrides.pop("da3_active", False),
        sfm_python=sfm_python,
        output_root=output_root,
        profile=overrides.pop("profile", ReconstructionQualityProfile.EFFICIENT),
        session_id=overrides.pop("session_id", SESSION),
        disk_free_bytes=overrides.pop("disk_free_bytes", 24 * 1024**3),
        minimum_free_disk_bytes=overrides.pop("minimum_free_disk_bytes", 8 * 1024**3),
        application_resource_guard_blocker=overrides.pop("application_resource_guard_blocker", None),
        **overrides,
    )


def test_i6_2a_ready_image_sequence_preflight_is_passive_and_explicit(tmp_path: Path) -> None:
    result = _build(tmp_path)

    assert result.status is OneStartPreflightStatus.READY
    assert result.ready is True
    assert result.blockers == ()
    assert result.workflow == "image_sequence"
    assert result.source_item_count == 4
    assert result.session_id == SESSION
    assert result.proposed_output_dir.endswith(SESSION)
    assert result.profile == ReconstructionQualityProfile.EFFICIENT.value
    assert result.camera_authority_owner == CANONICAL_CAMERA_AUTHORITY_OWNER
    assert result.execution_scope == ONE_START_EXECUTION_SCOPE
    assert result.production_auto_policy_scope == PRODUCTION_AUTO_POLICY_SCOPE
    assert result.production_auto_qualified is False
    assert result.production_auto_authorized is False
    assert result.dense_execution_authorized is False
    assert result.persistent_scene_promotion_authorized is False
    assert result.certified_metrology is False
    assert result.application_resource_guard_required is True


def test_i6_2a_recorded_video_is_supported_without_opening_backend(tmp_path: Path) -> None:
    video = tmp_path / "capture.mp4"
    video.write_bytes(b"video")
    result = _build(
        tmp_path,
        images=tmp_path / "unused",
        selection=_selection(video, workflow="recorded_video", item_count=1),
    )

    assert result.ready is True
    assert result.workflow == "recorded_video"
    assert result.source_path == str(video.resolve())
    assert result.source_item_count == 1


def test_i6_2a_live_source_is_blocked_from_offline_one_start(tmp_path: Path) -> None:
    source = tmp_path / "live"
    result = _build(
        tmp_path,
        selection=_selection(source, workflow="live_source", ready=True, item_count=1),
    )

    assert result.ready is False
    assert OneStartBlockerCode.UNSUPPORTED_WORKFLOW in result.blocker_codes


def test_i6_2a_stale_image_selection_is_revalidated_and_blocked(tmp_path: Path) -> None:
    images = _images(tmp_path, count=4)
    for path in sorted(images.glob("*.jpg"))[2:]:
        path.unlink()

    result = _build(tmp_path, images=images, selection=_selection(images, item_count=4))

    assert result.ready is False
    assert OneStartBlockerCode.IMAGE_COUNT_TOO_SMALL in result.blocker_codes
    assert result.source_item_count == 2


def test_i6_2a_missing_recorded_video_is_blocked_even_if_selection_was_ready(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mp4"
    result = _build(
        tmp_path,
        selection=_selection(missing, workflow="recorded_video", ready=True, item_count=1),
    )

    assert result.ready is False
    assert OneStartBlockerCode.SOURCE_PATH_MISSING in result.blocker_codes


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("controller_shutdown", OneStartBlockerCode.CONTROLLER_SHUTDOWN),
        ("live_active", OneStartBlockerCode.LIVE_RECONSTRUCTION_ACTIVE),
        ("offline_active", OneStartBlockerCode.OFFLINE_RECONSTRUCTION_ACTIVE),
        ("da3_active", OneStartBlockerCode.DA3_INTEGRATION_ACTIVE),
    ],
)
def test_i6_2a_active_or_shutdown_owners_block_start(
    tmp_path: Path, field: str, expected: OneStartBlockerCode
) -> None:
    result = _build(tmp_path, **{field: True})

    assert result.ready is False
    assert expected in result.blocker_codes


def test_i6_2a_missing_pycolmap_interpreter_blocks_before_execution(tmp_path: Path) -> None:
    result = _build(tmp_path, sfm_python=tmp_path / "missing-python")

    assert result.ready is False
    assert OneStartBlockerCode.PYCOLMAP_RUNTIME_MISSING in result.blocker_codes


def test_i6_2a_nonempty_output_collision_blocks_stale_result_reuse(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    proposed = output_root / SESSION
    proposed.mkdir(parents=True)
    (proposed / "old-result.json").write_text("{}", encoding="utf-8")

    result = _build(tmp_path, output_root=output_root)

    assert result.ready is False
    assert OneStartBlockerCode.OUTPUT_DIR_NOT_EMPTY in result.blocker_codes


def test_i6_2a_existing_mvs_disk_floor_is_reused_without_hidden_retuning(tmp_path: Path) -> None:
    floor = 8 * 1024**3
    result = _build(
        tmp_path,
        disk_free_bytes=floor - 1,
        minimum_free_disk_bytes=floor,
    )

    assert result.ready is False
    assert result.minimum_free_disk_bytes == floor
    assert result.disk_free_bytes == floor - 1
    assert OneStartBlockerCode.DISK_SAFETY_FLOOR_NOT_MET in result.blocker_codes



def test_i6_2a_nonexecutable_pycolmap_interpreter_is_blocked(tmp_path: Path) -> None:
    sfm = tmp_path / "pycolmap-python-noexec"
    sfm.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    sfm.chmod(0o644)

    result = _build(tmp_path, sfm_python=sfm)

    assert result.ready is False
    assert OneStartBlockerCode.PYCOLMAP_RUNTIME_NOT_EXECUTABLE in result.blocker_codes


def test_i6_2a_output_root_file_is_blocked_without_modification(tmp_path: Path) -> None:
    root = tmp_path / "not-a-directory"
    root.write_text("occupied", encoding="utf-8")

    result = _build(tmp_path, output_root=root)

    assert result.ready is False
    assert OneStartBlockerCode.OUTPUT_ROOT_INVALID in result.blocker_codes
    assert root.read_text(encoding="utf-8") == "occupied"

def test_i6_2a_application_resource_guard_failure_is_visible_and_fail_closed(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        application_resource_guard_blocker="Benchmark is active; stop it first.",
    )

    assert result.ready is False
    assert OneStartBlockerCode.APPLICATION_RESOURCE_CONFLICT in result.blocker_codes
    assert "Benchmark is active" in result.blocker_details


def test_i6_2a_invalid_session_identity_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="session_id"):
        _build(tmp_path, session_id="not-a-uuid")


def test_i6_2a_legal_runtime_transition_contract_matches_existing_offline_controller() -> None:
    assert offline_one_start_transition_allowed("idle", "starting") is True
    assert offline_one_start_transition_allowed("completed", "starting") is True
    assert offline_one_start_transition_allowed("failed", "starting") is True
    assert offline_one_start_transition_allowed("cancelled", "starting") is True
    assert offline_one_start_transition_allowed("starting", "running") is True
    assert offline_one_start_transition_allowed("starting", "failed") is True
    assert offline_one_start_transition_allowed("running", "cancelling") is True
    assert offline_one_start_transition_allowed("running", "completed") is True
    assert offline_one_start_transition_allowed("running", "failed") is True
    assert offline_one_start_transition_allowed("cancelling", "cancelled") is True
    assert offline_one_start_transition_allowed("cancelling", "failed") is False
    assert offline_one_start_transition_allowed("running", "starting") is False
    assert offline_one_start_transition_allowed("starting", "starting") is False


def test_i6_2a_as_dict_preserves_scientific_and_auto_boundaries(tmp_path: Path) -> None:
    payload = _build(tmp_path).as_dict()

    assert payload["authority"]["camera_authority_owner"] == "pycolmap_sfm"
    assert payload["authority"]["certified_metrology"] is False
    assert payload["execution"]["scope"] == "sparse_sfm_only"
    assert payload["execution"]["dense_execution_authorized"] is False
    assert payload["execution"]["persistent_scene_promotion_authorized"] is False
    assert payload["production_auto"] == {
        "policy_scope": "offline_shadow_only",
        "qualified": False,
        "authorized": False,
    }
