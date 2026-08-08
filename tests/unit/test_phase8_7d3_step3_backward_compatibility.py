from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_step3_keeps_step2_mode_as_default_and_does_not_enable_ui() -> None:
    runner = (ROOT / "scripts" / "run_gaussian_real_scene_benchmark.py").read_text(encoding="utf-8")
    assert '"--measure-performance"' in runner
    assert 'action="store_true"' in runner
    assert 'output / "step2_summary.json"' in runner
    assert 'output / "step3_summary.json"' in runner

    source = (ROOT / "core" / "reconstruction" / "gaussian_benchmarking.py").read_text(encoding="utf-8")
    assert "canonical_geometry_authoritative" in source
    assert "certified_metrology" in source

    for path in (ROOT / "app").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "step3_summary.json" not in text


def test_step3_does_not_modify_d2_gaussian_worker_sources() -> None:
    # Step 3 must extend only D3 benchmark workers; D2 workers remain the
    # qualification baseline and are protected by the D2 backward quick gate.
    d3_runner = (ROOT / "scripts" / "run_phase8_7d3_tests.sh").read_text(encoding="utf-8")
    assert "test_phase8_7d3_step3_backward_compatibility.py" in d3_runner
