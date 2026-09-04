from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_dynamic_scene_handling import (
    analyze_i5_5b_sfm_report,
    compare_i5_5b_sfm_outcomes,
    validate_i5_5a_for_i5_5b,
)
from tests.i5_5b_helpers import (
    clone,
    passing_i5_5a,
    passing_materialization_revalidation,
    synthetic_i55b_analysis,
    synthetic_i55b_sfm_report,
)


def test_i55b_validation_does_not_mutate_frozen_admission(tmp_path: Path) -> None:
    admission = passing_i5_5a(tmp_path)
    before = clone(admission)
    validate_i5_5a_for_i5_5b(admission)
    assert admission == before


def test_i55b_sfm_component_count_must_be_nonnegative(tmp_path: Path) -> None:
    admission = passing_i5_5a(tmp_path)
    report = synthetic_i55b_sfm_report(admission, "static_control")
    with pytest.raises(ReconstructionValidationError):
        analyze_i5_5b_sfm_report(
            report,
            scenario_plan=admission["stress_plan"]["static_control"],
            staged_names=[f"{i:04d}.png" for i in range(19)],
            sparse_component_count=-1,
        )


def test_i55b_comparison_rejects_missing_pycolmap_authority(tmp_path: Path) -> None:
    admission = passing_i5_5a(tmp_path)
    control = synthetic_i55b_analysis(admission, "static_control")
    stress = synthetic_i55b_analysis(admission, "dynamic_foreground_proxy")
    stress["authority"]["camera_pose_authority_owner"] = "other"
    with pytest.raises(ReconstructionValidationError):
        compare_i5_5b_sfm_outcomes(
            control=control,
            dynamic_proxy=stress,
            materialization_revalidation=passing_materialization_revalidation(),
        )
