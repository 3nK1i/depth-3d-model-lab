from __future__ import annotations

from pathlib import Path

from app.reconstruction_release_gate import (
    MAX_REAL_SCENE_DURATION_S,
    MIN_REAL_SCENE_DURATION_S,
    QUALIFIED_NERF_STEPS,
    RELEASE_GATE_SCHEMA,
    STATE_EXPORT_SCHEMA,
)

ROOT = Path(__file__).resolve().parents[2]


def test_phase96h_release_gate_is_validation_only() -> None:
    module = (ROOT / "app/reconstruction_release_gate.py").read_text(encoding="utf-8")
    runner = (ROOT / "scripts/run_phase9_6h_real_scene_gate.py").read_text(encoding="utf-8")
    forbidden = ("import torch", "import nerfstudio", "import gsplat", "depth_anything_3", "QProcess(")
    for token in forbidden:
        assert token not in module
        assert token not in runner
    assert "never starts a heavyweight backend" in runner


def test_phase96h_release_gate_bounds_are_explicit() -> None:
    assert RELEASE_GATE_SCHEMA == "eia.digital-twin.phase9-6h-release-gate.v1"
    assert STATE_EXPORT_SCHEMA == "eia.digital-twin.state-export.v1"
    assert (MIN_REAL_SCENE_DURATION_S, MAX_REAL_SCENE_DURATION_S) == (120.0, 180.0)
    assert QUALIFIED_NERF_STEPS == frozenset({64, 128, 256})


def test_phase96h_script_never_authorizes_tag_creation() -> None:
    runner = (ROOT / "scripts/run_phase9_6h_real_scene_gate.py").read_text(encoding="utf-8")
    assert '"v0_9_0_tag_authorized_by_script": False' in runner
    assert "git tag" not in runner
    assert "git push --tags" not in runner
