#!/usr/bin/env python3
"""Dependency-light static readiness smoke for the Phase 8.7G4 handoff gate."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PHASE = "8.7G4"
BANNED_HEAVY = {"torch", "pycolmap", "gsplat", "nerfstudio", "tinycudann", "lpips", "depth_anything_3"}
BANNED_CORE_UI = {"PyQt6", "app"}


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _reconstruction_enabled_in_source() -> bool:
    path = ROOT / "app/widgets/mode_tabs.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_ENABLED_MODES" for t in node.targets):
            if isinstance(node.value, (ast.Set, ast.Tuple, ast.List)):
                for item in node.value.elts:
                    if isinstance(item, ast.Attribute) and item.attr == "RECONSTRUCTION":
                        return True
    return False


def main() -> int:
    core_dir = ROOT / "core/reconstruction"
    heavy_violations: list[str] = []
    ui_violations: list[str] = []
    for path in sorted(core_dir.glob("*.py")):
        roots = _import_roots(path)
        heavy = roots & BANNED_HEAVY
        ui = roots & BANNED_CORE_UI
        if heavy:
            heavy_violations.append(f"{path.relative_to(ROOT)}:{','.join(sorted(heavy))}")
        if ui:
            ui_violations.append(f"{path.relative_to(ROOT)}:{','.join(sorted(ui))}")

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    production_dependency_drift = any(
        token in pyproject
        for token in ("gsplat", "nerfstudio", "tiny-cuda-nn", "tinycudann", "depth-anything-3", "pycolmap")
    )

    scripts = (
        "scripts/run_phase8_7g1_tests.sh",
        "scripts/run_phase8_7g2_tests.sh",
        "scripts/run_phase8_7g3_tests.sh",
        "scripts/run_phase8_7g4_tests.sh",
    )
    executable = {
        name: bool((ROOT / name).is_file() and (ROOT / name).stat().st_mode & stat.S_IXUSR)
        for name in scripts
    }

    from core.version import RELEASE_NAME, RELEASE_PHASE, __version__

    release_metadata_preserved = (
        __version__ == "0.7.5" and RELEASE_NAME == "Phase 7.5 Stable" and RELEASE_PHASE == 7.5
    )
    reconstruction_ui_enabled = _reconstruction_enabled_in_source()
    passed = bool(
        not heavy_violations
        and not ui_violations
        and not production_dependency_drift
        and all(executable.values())
        and release_metadata_preserved
        and not reconstruction_ui_enabled
    )
    payload = {
        "phase": PHASE,
        "pass": passed,
        "release_metadata_preserved": release_metadata_preserved,
        "heavy_import_violations": heavy_violations,
        "core_ui_import_violations": ui_violations,
        "production_dependency_drift": production_dependency_drift,
        "required_scripts_executable": executable,
        "reconstruction_ui_enabled": reconstruction_ui_enabled,
        "phase8_8_ui_handoff_target": True,
        "full_regression_required_before_handoff": True,
    }
    out = Path("/tmp/phase8_7g4_readiness")
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "summary.json"
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(f"Phase 8.7G4 readiness smoke: FAIL; summary={summary}")
    print("Phase 8.7G4 readiness contracts: PASS")
    print("release_metadata_preserved=True core_heavy_imports=False core_ui_imports=False")
    print("production_dependency_drift=False reconstruction_ui_enabled=False")
    print("full_regression_required_before_handoff=True phase8_8_ui_handoff_target=True")
    print(f"summary={summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
