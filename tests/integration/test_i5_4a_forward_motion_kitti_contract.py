from __future__ import annotations

from pathlib import Path

from core.reconstruction.i5_forward_motion_kitti import (
    build_i5_4a_forward_motion_kitti_admission,
    build_i5_4a_source_identity,
)
from tests.i5_4a_helpers import I53C_SHA, RECEIPT_54A_SHA, binding, kitti_source_dir, passing_i5_3c, passing_i5_4a_receipt


def test_i54a_binds_predecessor_source_and_window_without_execution(tmp_path: Path) -> None:
    identity = build_i5_4a_source_identity(kitti_source_dir(tmp_path))
    result = build_i5_4a_forward_motion_kitti_admission(
        i5_3c_result=passing_i5_3c(tmp_path),
        i5_3c_binding=binding("/evidence/i5_3c.json", I53C_SHA),
        source_identity=identity,
        source_label="KITTI seq00 image_2",
        kitti_style_forward_motion_confirmation=True,
        admission_receipt=passing_i5_4a_receipt(),
        admission_receipt_binding=binding("/evidence/i5_4a_receipt.json", RECEIPT_54A_SHA),
    )
    assert result["source_identity"]["asset_set_sha256"] == identity["asset_set_sha256"]
    assert result["stress_plan"]["forward_motion_window"]["names"] == identity["ordered_names"][:60]
    assert result["scope"]["sfm_rerun"] is False
    assert result["scope"]["images_transformed"] is False


def test_i54a_admission_carries_exact_i53c_sha_binding(tmp_path: Path) -> None:
    result = build_i5_4a_forward_motion_kitti_admission(
        i5_3c_result=passing_i5_3c(tmp_path),
        i5_3c_binding=binding("/evidence/i5_3c.json", I53C_SHA),
        source_identity=build_i5_4a_source_identity(kitti_source_dir(tmp_path)),
        source_label="KITTI seq00 image_2",
        kitti_style_forward_motion_confirmation=True,
        admission_receipt=passing_i5_4a_receipt(),
        admission_receipt_binding=binding("/evidence/i5_4a_receipt.json", RECEIPT_54A_SHA),
    )
    assert result["evidence_bindings"]["i5_3c_closure"]["sha256"] == I53C_SHA
