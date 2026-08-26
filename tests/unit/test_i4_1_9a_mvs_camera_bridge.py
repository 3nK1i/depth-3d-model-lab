from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_camera_bridge import (
    CameraBridgeInputs,
    CameraBridgeView,
    MVS_CAMERA_BRIDGE_EXPECTED_GLOBAL_VIEWS,
    MVS_CAMERA_BRIDGE_EXPECTED_LOCALIZED_VIEWS,
    MVS_CAMERA_BRIDGE_EXPECTED_TOTAL_VIEWS,
    _point3d_id,
    _write_derived_text_model,
)


def _pose(x: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[0, 3] = x
    return result


def test_house_bridge_counts_are_explicit_not_a_permanent_40_view_cap() -> None:
    assert MVS_CAMERA_BRIDGE_EXPECTED_GLOBAL_VIEWS == 40
    assert MVS_CAMERA_BRIDGE_EXPECTED_LOCALIZED_VIEWS == 28
    assert MVS_CAMERA_BRIDGE_EXPECTED_TOTAL_VIEWS == 68


def test_only_authoritative_pycolmap_landmark_ids_are_accepted() -> None:
    assert _point3d_id("pycolmap-landmark-123") == 123
    with pytest.raises(ReconstructionValidationError):
        _point3d_id("synthetic-point-123")


def test_localized_view_requires_fixed_pose_and_existing_landmark_observations(tmp_path: Path) -> None:
    image = tmp_path / "candidate.jpg"
    image.write_bytes(b"image")
    view = CameraBridgeView(
        source_sequence_index=1,
        provenance="localized_extra_observation",
        image_name=image.name,
        image_path=str(image),
        camera_id=1,
        T_reconstruction_from_camera=_pose(1.0),
        matched_point3d_ids=(7,),
        matched_feature_xy=((10.0, 20.0),),
    )
    assert view.backend_image_id is None
    assert view.matched_point3d_ids == (7,)


def test_text_bridge_preserves_native_records_and_appends_only_extra_tracks(tmp_path: Path) -> None:
    native = tmp_path / "native"
    native.mkdir()
    (native / "cameras.txt").write_text("# cameras\n1 SIMPLE_RADIAL 100 80 90 50 40 0.01\n", encoding="utf-8")
    (native / "images.txt").write_text(
        "# images\n"
        "1 1 0 0 0 0 0 0 1 global0.jpg\n10 20 7\n"
        "2 1 0 0 0 -1 0 0 1 global1.jpg\n11 21 7\n",
        encoding="utf-8",
    )
    (native / "points3D.txt").write_text(
        "# points\n7 0 0 3 1 2 3 0.5 1 0 2 0\n",
        encoding="utf-8",
    )
    global0 = tmp_path / "global0.jpg"; global0.write_bytes(b"g0")
    global1 = tmp_path / "global1.jpg"; global1.write_bytes(b"g1")
    extra = tmp_path / "candidate_00000002.jpg"; extra.write_bytes(b"e")
    inputs = CameraBridgeInputs(
        i4_1_7_dir=tmp_path,
        i4_1_7_summary=tmp_path / "summary.json",
        orbit_diagnostics=tmp_path / "diag.npz",
        sfm_report=tmp_path / "sfm.json",
        native_sparse_model=native,
        localization_dir=tmp_path,
        selected_model_id=0,
        shared_camera_id=1,
        views=(
            CameraBridgeView(0, "global_sfm_anchor", "global0.jpg", str(global0), 1, _pose(0.0), backend_image_id=1),
            CameraBridgeView(1, "global_sfm_anchor", "global1.jpg", str(global1), 1, _pose(1.0), backend_image_id=2),
            CameraBridgeView(2, "localized_extra_observation", extra.name, str(extra), 1, _pose(2.0), matched_point3d_ids=(7,), matched_feature_xy=((12.0, 22.0),)),
        ),
        camera_model_payload={"backend_camera_id": 1},
        native_workspace_root=tmp_path,
    )
    # This helper normally receives the House 40+28 set. Patch the module-level
    # constant only within this bounded unit test so the serialization contract
    # can be exercised on a tiny model.
    import core.reconstruction.mvs_camera_bridge as bridge
    old = bridge.MVS_CAMERA_BRIDGE_EXPECTED_GLOBAL_VIEWS
    bridge.MVS_CAMERA_BRIDGE_EXPECTED_GLOBAL_VIEWS = 2
    try:
        assigned, stats = _write_derived_text_model(
            native_text_dir=native,
            output_sparse_dir=tmp_path / "derived",
            inputs=inputs,
        )
    finally:
        bridge.MVS_CAMERA_BRIDGE_EXPECTED_GLOBAL_VIEWS = old
    assert assigned == {2: 3}
    assert stats["derived_registered_images"] == 3
    images = (tmp_path / "derived/images.txt").read_text(encoding="utf-8")
    points = (tmp_path / "derived/points3D.txt").read_text(encoding="utf-8")
    assert "1 1 0 0 0 0 0 0 1 global0.jpg" in images
    assert "3 " in images and extra.name in images
    assert points.strip().endswith("1 0 2 0 3 0")
