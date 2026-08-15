from __future__ import annotations

import ast
from pathlib import Path

from core.reconstruction import (
    NerfBackendKind,
    NerfInteractiveRuntimeState,
    nerf_backend_provenance,
)

ROOT = Path(__file__).resolve().parents[2]
HEAVY = {"nerfstudio", "tinycudann"}


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


def test_5a_production_sources_do_not_directly_import_nerfstudio_or_tinycudann() -> None:
    assert _direct_heavy_imports(ROOT / "core") == []
    assert _direct_heavy_imports(ROOT / "app") == []


def test_5a_production_dependencies_do_not_install_nerf_heavy_runtime() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "nerfstudio" not in pyproject
    assert "tinycudann" not in pyproject
    assert "tiny-cuda-nn" not in pyproject


def test_5a_reuses_phase8_process_group_safety_and_preserves_permissive_appearance_policy() -> None:
    lifecycle = (ROOT / "scripts" / "nerf_backend_lifecycle.py").read_text(encoding="utf-8")
    assert "start_new_session=True" in lifecycle
    assert "os.killpg" in lifecycle
    assert "signal.SIGTERM" in lifecycle
    assert "signal.SIGKILL" in lifecycle

    provenance = nerf_backend_provenance(NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL)
    assert provenance.license_name == "Apache-2.0"
    assert provenance.commercial_use_permitted_by_backend_license is True
    assert provenance.runtime_dependency_name == "tiny-cuda-nn"
    assert provenance.runtime_dependency_license == "BSD-3-Clause"
    assert NerfInteractiveRuntimeState.READY.value == "ready"
