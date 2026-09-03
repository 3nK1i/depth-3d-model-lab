from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_forward_motion_kitti import (
    build_i5_4a_forward_motion_kitti_admission,
    build_i5_4a_source_identity,
    validate_i5_3c_for_i5_4,
)
from tests.i5_4a_helpers import I53C_SHA, RECEIPT_54A_SHA, binding, clone, kitti_source_dir, passing_i5_3c, passing_i5_4a_receipt


def test_i54a_rejects_forged_i53c_authority(tmp_path: Path) -> None:
    closure = clone(passing_i5_3c(tmp_path))
    closure["authority"]["camera_pose_authority_owner"] = "dense_provider"
    with pytest.raises(ReconstructionValidationError):
        validate_i5_3c_for_i5_4(closure)


def test_i54a_rejects_i53c_without_i54_authorization(tmp_path: Path) -> None:
    closure = clone(passing_i5_3c(tmp_path))
    closure["decision"]["i5_4_authorized"] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i5_3c_for_i5_4(closure)


def test_i54a_rejects_source_identity_tamper(tmp_path: Path) -> None:
    identity = build_i5_4a_source_identity(kitti_source_dir(tmp_path))
    identity["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ReconstructionValidationError):
        build_i5_4a_forward_motion_kitti_admission(
            i5_3c_result=passing_i5_3c(tmp_path),
            i5_3c_binding=binding("/evidence/i5_3c.json", I53C_SHA),
            source_identity=identity,
            source_label="KITTI seq00 image_2",
            kitti_style_forward_motion_confirmation=True,
            admission_receipt=passing_i5_4a_receipt(),
            admission_receipt_binding=binding("/evidence/i5_4a_receipt.json", RECEIPT_54A_SHA),
        )
