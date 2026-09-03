from __future__ import annotations

from pathlib import Path

from core.reconstruction.i5_low_overlap_weak_parallax import analyze_i5_2b_sfm_report, compare_i5_2b_stress_geometry
from tests.i5_2b_helpers import passing_i5_2a, synthetic_sfm_report

ROOT = Path(__file__).resolve().parents[2]


def test_i52b_runner_reuses_current_production_native_sfm_path() -> None:
    text = (ROOT / "scripts/run_i5_2b_low_overlap_weak_parallax_replay.py").read_text(encoding="utf-8")
    assert "run_reconstruction_pycolmap_smoke.py" in text
    assert "--images" in text
    assert "--matching" in text and "exhaustive" in text
    assert "import pycolmap" not in text
    assert "run_i4_3g2b_mvs_da3_scenario.py" not in text
    assert "dense_provider" not in text.lower()


def test_i52b_contract_preserves_three_frozen_slices_and_same_matching_policy(tmp_path: Path) -> None:
    admission = passing_i5_2a(tmp_path)
    assert admission["stress_plan"]["control"]["image_count"] == 19
    assert admission["stress_plan"]["low_overlap_proxy"]["image_count"] == 7
    assert admission["stress_plan"]["weak_parallax_proxy"]["image_count"] == 7
    assert admission["stress_plan"]["execution_contract"]["matching_strategy_locked_for_like_for_like_replay"] == "exhaustive"


def test_i52b_geometry_analysis_survives_partial_registration_without_inventing_full_scope(tmp_path: Path) -> None:
    admission = passing_i5_2a(tmp_path)
    report = synthetic_sfm_report(
        admission,
        "low_overlap_proxy",
        camera_step=2.0,
        common_landmarks_per_adjacent_pair=10,
        registered_count=5,
        quality_status="accepted",
    )
    analysis = analyze_i5_2b_sfm_report(report, scenario_plan=admission["stress_plan"]["low_overlap_proxy"])
    assert analysis["input_images"] == 7
    assert analysis["registered_images"] == 5
    assert analysis["partial_registration_preserved"] is True
    assert analysis["full_input_scope_registered"] is False


def test_i52b_comparison_is_scale_invariant_not_metric(tmp_path: Path) -> None:
    admission = passing_i5_2a(tmp_path)
    analyses = {}
    for key, step, common in (
        ("control", 1.0, 40),
        ("low_overlap_proxy", 2.0, 12),
        ("weak_parallax_proxy", 0.2, 35),
    ):
        analyses[key] = analyze_i5_2b_sfm_report(
            synthetic_sfm_report(admission, key, camera_step=step, common_landmarks_per_adjacent_pair=common),
            scenario_plan=admission["stress_plan"][key],
        )
    comparison = compare_i5_2b_stress_geometry(
        control=analyses["control"],
        low_overlap=analyses["low_overlap_proxy"],
        weak_parallax=analyses["weak_parallax_proxy"],
    )
    assert comparison["comparison_gauge"] == "per_scenario_arbitrary_sfm_gauge"
    assert comparison["cross_scenario_absolute_baseline_comparison_permitted"] is False
    assert comparison["scale_invariant_metrics_only"] is True
