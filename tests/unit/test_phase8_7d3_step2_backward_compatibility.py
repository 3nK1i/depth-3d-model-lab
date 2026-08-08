from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_step2_keeps_lpips_out_of_production_core_and_d2_workers() -> None:
    for root in (ROOT / "core" / "reconstruction", ROOT / "scripts" / "gaussian_workers"):
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "import lpips" not in text
            assert "from lpips" not in text


def test_step2_metrics_worker_is_external_and_pinned() -> None:
    path = ROOT / "scripts" / "gaussian_benchmark_workers" / "metrics_worker.py"
    text = path.read_text(encoding="utf-8")
    assert 'LPIPS_PACKAGE_VERSION = "0.1.4"' in text
    assert 'LPIPS_METRIC_VERSION = "0.1"' in text
    assert 'LPIPS_NETWORK = "alex"' in text
    assert "canonical_geometry_authoritative" in text
    assert "certified_metrology" in text


def test_graphdeco_benchmark_fails_closed_without_explicit_test_split_support() -> None:
    path = ROOT / "scripts" / "gaussian_benchmark_workers" / "graphdeco_worker.py"
    text = path.read_text(encoding="utf-8")
    assert '"test.txt"' in text
    assert '"--eval"' in text
    assert '"--llffhold"' not in text
    assert '"--skip_train"' in text
    assert "llffhold=0" in text
    assert "held-out validation would risk training-view leakage" in text
    assert '(sparse_dir / "test.txt").write_text' in text
    assert '(target / "test.txt").write_text' not in text


def test_production_dependencies_are_not_changed_by_step2_overlay() -> None:
    # Step 2 uses an isolated metrics environment instead of adding LPIPS to pyproject.toml.
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "lpips" not in pyproject
