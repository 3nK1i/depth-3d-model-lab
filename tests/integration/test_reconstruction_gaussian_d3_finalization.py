from pathlib import Path


def test_finalization_runner_preserves_external_backend_boundary() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts" / "run_gaussian_d3_finalization.py").read_text(encoding="utf-8")
    assert "run_gaussian_backend_qualification.py" in text
    assert "run_gaussian_heldout_quality_smoke.py" in text
    assert "run_gaussian_performance_smoke.py" in text
    assert "run_gaussian_backend_robustness_smoke.py" in text
    assert "import gsplat" not in text
    assert "import gaussian_renderer" not in text
