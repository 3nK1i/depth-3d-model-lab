from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_forward_motion_kitti import analyze_i5_4b_sfm_report, validate_i5_4a_for_i5_4b
from tests.i5_4b_helpers import clone, passing_i5_4a, synthetic_i54b_sfm_report


def test_i54b_rejects_window_hash_drift(tmp_path: Path) -> None:
    admission = clone(passing_i5_4a(tmp_path))
    admission["qualification"]["window_asset_set_sha256"] = "0" * 64
    with pytest.raises(ReconstructionValidationError):
        validate_i5_4a_for_i5_4b(admission)


def test_i54b_rejects_noncontiguous_window_drift(tmp_path: Path) -> None:
    admission = clone(passing_i5_4a(tmp_path))
    admission["stress_plan"]["forward_motion_window"]["contiguous"] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i5_4a_for_i5_4b(admission)


def test_i54b_rejects_workspace_identity_drift(tmp_path: Path) -> None:
    admission = passing_i5_4a(tmp_path)
    report = synthetic_i54b_sfm_report(admission)
    report["sfm_result"]["workspace"]["images"][0]["source_locator"] = "/scenario/not-frozen.png"
    report["sfm_result"]["workspace"]["images"][0]["staged_image_path"] = "images/not-frozen.png"
    with pytest.raises(ReconstructionValidationError):
        analyze_i5_4b_sfm_report(report, scenario_plan=admission["stress_plan"]["forward_motion_window"])


def test_i54b_handles_sparse_adjacent_registration_without_fabricating_motion(tmp_path: Path) -> None:
    admission = passing_i5_4a(tmp_path)
    report = synthetic_i54b_sfm_report(admission, registered_count=1)
    analysis = analyze_i5_4b_sfm_report(report, scenario_plan=admission["stress_plan"]["forward_motion_window"])
    assert analysis["geometry"]["camera_motion"]["adjacent_registered_motion_pairs"] == 0
    assert analysis["geometry"]["camera_motion"]["authoritative_forward_motion_geometry_proven"] is False
