"""Static final-readiness invariants for Phase 8.7G4."""
from __future__ import annotations

import ast
from pathlib import Path
import stat

from core.version import RELEASE_NAME, RELEASE_PHASE, __version__

ROOT = Path(__file__).resolve().parents[2]
BANNED_HEAVY = {"torch", "pycolmap", "gsplat", "nerfstudio", "tinycudann", "lpips", "depth_anything_3"}


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_g4_preserves_current_release_identity_after_phase8_release() -> None:
    assert __version__ == "0.9.0"
    assert RELEASE_NAME == "Phase 9 Digital Twin Stable"
    assert RELEASE_PHASE == 9.0


def test_reconstruction_core_has_no_direct_heavy_backend_imports() -> None:
    violations = {}
    for path in sorted((ROOT / "core/reconstruction").glob("*.py")):
        bad = _import_roots(path) & BANNED_HEAVY
        if bad:
            violations[str(path.relative_to(ROOT))] = sorted(bad)
    assert violations == {}


def test_reconstruction_core_remains_ui_independent() -> None:
    violations = {}
    for path in sorted((ROOT / "core/reconstruction").glob("*.py")):
        bad = _import_roots(path) & {"PyQt6", "app"}
        if bad:
            violations[str(path.relative_to(ROOT))] = sorted(bad)
    assert violations == {}


def test_production_pyproject_does_not_absorb_external_reconstruction_runtimes() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    for token in ("pycolmap", "gsplat", "nerfstudio", "tiny-cuda-nn", "tinycudann", "depth-anything-3"):
        assert token not in text


def test_reconstruction_mode_remains_disabled_until_phase_88() -> None:
    path = ROOT / "app/widgets/mode_tabs.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    enabled_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_ENABLED_MODES" for t in node.targets):
            assert isinstance(node.value, ast.Set)
            for item in node.value.elts:
                if isinstance(item, ast.Attribute):
                    enabled_names.add(item.attr)
    assert "RECONSTRUCTION" not in enabled_names
    assert {"LIVE_3D", "INSPECTION", "STABLE_VIDEO", "SNAPSHOT", "BENCHMARK"} <= enabled_names


def test_g1_through_g4_gate_scripts_are_executable() -> None:
    for relative in (
        "scripts/run_phase8_7g1_tests.sh",
        "scripts/run_phase8_7g2_tests.sh",
        "scripts/run_phase8_7g3_tests.sh",
        "scripts/run_phase8_7g4_tests.sh",
        "scripts/run_phase8_7g4_full_regression.py",
        "scripts/run_reconstruction_g4_readiness_smoke.py",
        "scripts/run_reconstruction_g4_real_e2e.py",
    ):
        path = ROOT / relative
        assert path.is_file(), relative
        assert path.stat().st_mode & stat.S_IXUSR, relative
