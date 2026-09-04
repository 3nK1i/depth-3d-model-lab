from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_dynamic_scene_handling import build_i5_5a_source_identity, build_i5_5a_stress_plan, validate_i5_5a_source_identity
from tests.i5_5a_helpers import castle_source_dir, clone


def test_i55a_rejects_tampered_source_aggregate(tmp_path: Path) -> None:
    identity = build_i5_5a_source_identity(castle_source_dir(tmp_path))
    bad = clone(identity)
    bad["asset_set_sha256"] = "0" * 64
    with pytest.raises(ReconstructionValidationError):
        validate_i5_5a_source_identity(bad)


def test_i55a_rejects_tampered_file_record(tmp_path: Path) -> None:
    identity = build_i5_5a_source_identity(castle_source_dir(tmp_path))
    bad = clone(identity)
    bad["files"][4]["sha256"] = "0" * 64
    with pytest.raises(ReconstructionValidationError):
        validate_i5_5a_source_identity(bad)


def test_i55a_non_image_metadata_does_not_change_image_count(tmp_path: Path) -> None:
    root = castle_source_dir(tmp_path)
    (root / "notes.json").write_text("{}", encoding="utf-8")
    identity = build_i5_5a_source_identity(root)
    assert identity["image_count"] == 19


def test_i55a_stress_recipe_changes_when_source_identity_changes(tmp_path: Path) -> None:
    first_root = castle_source_dir(tmp_path / "a")
    second_root = castle_source_dir(tmp_path / "b")
    (second_root / "0009.jpg").write_bytes(b"different donor bytes")
    a = build_i5_5a_stress_plan(build_i5_5a_source_identity(first_root))["dynamic_foreground_proxy"]["recipe_sha256"]
    b = build_i5_5a_stress_plan(build_i5_5a_source_identity(second_root))["dynamic_foreground_proxy"]["recipe_sha256"]
    assert a != b
