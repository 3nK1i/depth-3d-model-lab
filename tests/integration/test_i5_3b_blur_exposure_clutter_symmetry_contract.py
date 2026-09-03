from __future__ import annotations

from pathlib import Path


def test_i53b_runner_uses_production_native_pycolmap_path() -> None:
    text = Path("scripts/run_i5_3b_blur_exposure_clutter_symmetry_replay.py").read_text(encoding="utf-8")
    assert "run_reconstruction_pycolmap_smoke.py" in text
    assert '"exhaustive"' in text
    assert "--max-keyframes" in text
    assert "--image-sequence-fps" in text


def test_i53b_runner_materializes_exact_frozen_recipes_without_tuning() -> None:
    text = Path("scripts/run_i5_3b_blur_exposure_clutter_symmetry_replay.py").read_text(encoding="utf-8")
    assert "apply_i5_3b_transform" in text
    assert "stress_plan" in text
    assert "retry" not in text.lower() or "retry_until_quality_pass" in text


def test_i53b_documentation_preserves_proxy_and_authority_boundaries() -> None:
    text = Path("docs/phase_history/I5_3B_CURRENT_FROZEN_FOUNDATION_BLUR_EXPOSURE_CLUTTER_SYMMETRY_REPLAY.md").read_text(encoding="utf-8")
    for phrase in (
        "PyCOLMAP",
        "no threshold relaxation",
        "not natural-scene ground truth",
        "I5.3C",
        "I5.4",
    ):
        assert phrase in text


def test_i53b_runner_preserves_known_pre_sfm_ingestion_rejection() -> None:
    text = Path("scripts/run_i5_3b_blur_exposure_clutter_symmetry_replay.py").read_text(encoding="utf-8")
    assert "Fewer than three valid keyframes survived ingestion." in text
    assert "analyze_i5_3b_pre_sfm_ingestion_rejection" in text
    assert "not a recognized pre-SfM ingestion rejection" in text
