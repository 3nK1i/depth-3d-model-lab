from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "app" / "reconstruction_controller.py"


def _controller_text() -> str:
    return CONTROLLER.read_text(encoding="utf-8")


def test_i6_1b_controller_declares_workspace_signal_and_public_snapshot_property() -> None:
    tree = ast.parse(_controller_text())
    controller = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ReconstructionController"
    )

    signal_names = {
        target.id
        for node in controller.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "pyqtSignal"
    }
    property_names = {
        node.name
        for node in controller.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(isinstance(item, ast.Name) and item.id == "property" for item in node.decorator_list)
    }

    assert "offline_workspace_changed" in signal_names
    assert "offline_workspace_snapshot" in property_names


def test_i6_1b_controller_uses_i6_1a_builder_without_importing_auto_policy() -> None:
    text = _controller_text()

    assert "from app.reconstruction_offline_workspace import" in text
    assert "build_offline_reconstruction_workspace_snapshot" in text
    assert "OfflineReconstructionWorkspaceSnapshot" in text
    assert "i5_auto" not in text
    assert "AutoPolicy" not in text


def test_i6_1b_controller_has_deduplicated_workspace_emission_and_passive_event_cache() -> None:
    text = _controller_text()

    assert "_latest_offline_workspace_event" in text
    assert "_last_emitted_offline_workspace_snapshot" in text
    assert "def _emit_offline_workspace_changed" in text
    assert "def _on_offline_event_received" in text
    assert "def _on_offline_state_changed" in text
