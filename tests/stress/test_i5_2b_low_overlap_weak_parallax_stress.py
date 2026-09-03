from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from core.reconstruction.i5_low_overlap_weak_parallax import analyze_i5_2b_sfm_report, compare_i5_2b_stress_geometry
from tests.i5_2b_helpers import passing_i5_2a, synthetic_sfm_report


def _scale_report(report: dict, factor: float) -> dict:
    out = deepcopy(report)
    for camera in out["sfm_result"]["cameras"]:
        T = camera["T_reconstruction_from_camera"]
        T[0][3] *= factor
        T[1][3] *= factor
        T[2][3] *= factor
    for landmark in out["sfm_result"]["landmarks"]:
        landmark["position_reconstruction"] = [float(v) * factor for v in landmark["position_reconstruction"]]
    return out


@pytest.mark.parametrize("factor", [0.01, 0.1, 1.0, 10.0, 100.0])
def test_i52b_parallax_and_baseline_ratio_are_gauge_scale_invariant(tmp_path: Path, factor: float) -> None:
    admission = passing_i5_2a(tmp_path)
    report = synthetic_sfm_report(admission, "control", camera_step=1.0, common_landmarks_per_adjacent_pair=30)
    baseline = analyze_i5_2b_sfm_report(report, scenario_plan=admission["stress_plan"]["control"])
    scaled = analyze_i5_2b_sfm_report(_scale_report(report, factor), scenario_plan=admission["stress_plan"]["control"])
    assert scaled["geometry"]["parallax"]["maximum_pair_parallax_degrees_quantiles"] == pytest.approx(
        baseline["geometry"]["parallax"]["maximum_pair_parallax_degrees_quantiles"]
    )
    assert scaled["geometry"]["parallax"]["maximum_baseline_to_range_ratio_quantiles"] == pytest.approx(
        baseline["geometry"]["parallax"]["maximum_baseline_to_range_ratio_quantiles"]
    )


def test_i52b_directional_classification_is_deterministic_over_repeated_analysis(tmp_path: Path) -> None:
    admission = passing_i5_2a(tmp_path)
    reports = {
        "control": synthetic_sfm_report(admission, "control", camera_step=1.0, common_landmarks_per_adjacent_pair=40),
        "low": synthetic_sfm_report(admission, "low_overlap_proxy", camera_step=2.0, common_landmarks_per_adjacent_pair=12),
        "weak": synthetic_sfm_report(admission, "weak_parallax_proxy", camera_step=0.18, common_landmarks_per_adjacent_pair=35),
    }
    outputs = []
    for _ in range(25):
        outputs.append(compare_i5_2b_stress_geometry(
            control=analyze_i5_2b_sfm_report(reports["control"], scenario_plan=admission["stress_plan"]["control"]),
            low_overlap=analyze_i5_2b_sfm_report(reports["low"], scenario_plan=admission["stress_plan"]["low_overlap_proxy"]),
            weak_parallax=analyze_i5_2b_sfm_report(reports["weak"], scenario_plan=admission["stress_plan"]["weak_parallax_proxy"]),
        ))
    assert all(item == outputs[0] for item in outputs[1:])
    assert outputs[0]["semantic_stress_revalidation_complete"] is True
