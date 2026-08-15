from __future__ import annotations

import ast
from pathlib import Path

from core.reconstruction import (
    GaussianBackendKind,
    GaussianInteractiveRuntimeState,
    gaussian_backend_provenance,
)

ROOT = Path(__file__).resolve().parents[2]
HEAVY = {"gsplat", "diff_gaussian_rasterization", "simple_knn"}


def _direct_heavy_imports(root: Path) -> list[str]:
    hits: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".", 1)[0] in HEAVY:
                    hits.append(f"{path.relative_to(ROOT)}:{name}")
    return hits


def test_4a_production_sources_do_not_directly_import_heavy_gaussian_backends() -> None:
    assert _direct_heavy_imports(ROOT / "core") == []
    assert _direct_heavy_imports(ROOT / "app") == []


def test_4a_production_dependencies_do_not_install_gaussian_heavy_runtime() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    for package in ("gsplat", "diff-gaussian-rasterization", "simple-knn"):
        assert package not in pyproject


def test_4a_reuses_phase8_process_group_safety_and_keeps_graphdeco_research_only() -> None:
    lifecycle = (ROOT / "scripts" / "gaussian_backend_lifecycle.py").read_text(encoding="utf-8")
    assert "start_new_session=True" in lifecycle
    assert "os.killpg" in lifecycle
    assert "signal.SIGTERM" in lifecycle
    assert "signal.SIGKILL" in lifecycle

    graphdeco = gaussian_backend_provenance(GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL)
    gsplat = gaussian_backend_provenance(GaussianBackendKind.GSPLAT_EXTERNAL)
    assert graphdeco.usage_scope.value == "research_evaluation_only"
    assert graphdeco.commercial_use_permitted_by_backend_license is False
    assert gsplat.usage_scope.value == "permissive_general_use"
    assert gsplat.commercial_use_permitted_by_backend_license is True

    # Import/public-contract sanity: Phase 9.4A adds readiness states only and
    # does not enable training/rendering in-process.
    assert GaussianInteractiveRuntimeState.READY.value == "ready"
