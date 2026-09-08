from __future__ import annotations

import ast
from pathlib import Path

from core.reconstruction import (
    NerfBackendKind,
    NerfSceneRole,
    nerf_backend_runtime_pin,
)

ROOT = Path(__file__).resolve().parents[2]



def _module_scope_import_roots(path: Path) -> set[str]:
    """Return imports reachable at module import time, excluding function/class bodies."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return roots


def test_e4_preserves_e2_runtime_pin_and_appearance_authority() -> None:
    pin = nerf_backend_runtime_pin()
    assert pin.backend_kind is NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL
    assert pin.release_tag == "v1.1.5"
    assert pin.revision_prefix == "6b60855"
    assert NerfSceneRole.APPEARANCE_ONLY.value == "appearance_only"


def test_e4_does_not_add_heavy_nerf_runtime_to_production_manifest() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    for token in ("nerfstudio", "tiny-cuda-nn", "tinycudann", "lpips"):
        assert token not in text


def test_e4_production_core_has_no_eager_heavy_runtime_imports() -> None:
    banned = {"nerfstudio", "tinycudann", "torch", "lpips"}
    violations = {}
    for path in sorted((ROOT / "core/reconstruction").glob("*.py")):
        bad = _module_scope_import_roots(path) & banned
        if bad:
            violations[str(path.relative_to(ROOT))] = sorted(bad)
    assert violations == {}


def test_e3_quality_runner_and_shared_metrics_worker_remain_unchanged_in_role() -> None:
    e3 = (ROOT / "scripts/run_nerf_real_scene_validation.py").read_text(encoding="utf-8")
    metrics = (ROOT / "scripts/gaussian_benchmark_workers/metrics_worker.py").read_text(encoding="utf-8")
    assert "Phase 8.7E3 real-scene held-out Nerfacto quality validation" in e3
    assert "Backend-neutral PSNR/SSIM/LPIPS evaluator" in metrics
    assert '"gaussian_benchmark_workers" / "metrics_worker.py"' in e3


def test_e4_scope_does_not_claim_whole_engine_or_whole_app_gpu_hardening() -> None:
    perf = (ROOT / "scripts/run_nerf_performance_hardening.py").read_text(encoding="utf-8")
    robust = (ROOT / "scripts/run_nerf_backend_robustness.py").read_text(encoding="utf-8")
    assert '"8.7G"' in perf and '"8.9D"' in perf
    assert '"8.7G"' in robust
    assert "reconstruction_ui_enabled" in perf
    assert "reconstruction_ui_enabled" in robust
