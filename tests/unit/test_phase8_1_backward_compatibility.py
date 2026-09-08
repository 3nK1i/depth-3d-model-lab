from __future__ import annotations

import ast
from pathlib import Path

from core.contracts import CameraIntrinsics, FrameReference, RigidTransform3D
from core.hardware.profiles import ComputeProfile, resolve_profile
from core.reconstruction import RECONSTRUCTION_SCHEMA_VERSION, ReconstructionSession
from core.runtime.session import SessionTracker
from core.version import RELEASE_NAME, __version__



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


def test_phase81_backward_contracts_survive_current_release_identity() -> None:
    assert __version__ == "0.9.0"
    assert RELEASE_NAME == "Phase 9 Digital Twin Stable"
    assert CameraIntrinsics is not None
    assert FrameReference is not None
    assert RigidTransform3D is not None
    assert ComputeProfile is not None
    assert resolve_profile is not None
    assert SessionTracker is not None
    assert ReconstructionSession is not None
    assert RECONSTRUCTION_SCHEMA_VERSION == 1


def test_phase81_keeps_reconstruction_mode_disabled() -> None:
    source = Path("app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    enabled_line = next(line for line in source.splitlines() if line.startswith("_ENABLED_MODES"))
    assert "AppMode.RECONSTRUCTION" not in enabled_line


def test_phase81_foundation_has_no_eager_heavy_backend_imports() -> None:
    banned = {"open3d", "pycolmap", "torch", "PyQt6"}
    violations = {}
    for path in sorted(Path("core/reconstruction").glob("*.py")):
        bad = _module_scope_import_roots(path) & banned
        if bad:
            violations[path.name] = sorted(bad)
    assert violations == {}
