from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_b1_page_has_no_heavy_backend_or_test_fixture_imports() -> None:
    source = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    forbidden = ("pycolmap", "gsplat", "nerfstudio", "tinycudann", "depth_anything_3", "tests")
    assert not any(name.startswith(forbidden) for name in imports)


def test_b1_page_binding_contains_no_reconstruction_start_calls() -> None:
    source = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    page = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ReconstructionPage"
    )
    binding = next(
        node for node in page.body if isinstance(node, ast.FunctionDef) and node.name == "bind_controller"
    )
    binding_source = ast.get_source_segment(source, binding) or ""
    forbidden_calls = ("start_live(", "start_offline(", "start_selected_", "pause_live(", "resume_live(")
    assert not any(call in binding_source for call in forbidden_calls)


def test_b1_keeps_reconstruction_out_of_enabled_modes() -> None:
    source = (ROOT / "app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    enabled = source.split("_ENABLED_MODES =", 1)[1].split("}", 1)[0]
    assert "AppMode.RECONSTRUCTION" not in enabled


def test_b1_does_not_add_reconstruction_routing_branch() -> None:
    source = (ROOT / "app/main_window.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main_window = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow"
    )
    handler = next(
        node for node in main_window.body if isinstance(node, ast.FunctionDef) and node.name == "_on_mode_changed"
    )
    handler_source = ast.get_source_segment(source, handler) or ""
    assert "AppMode.RECONSTRUCTION" not in handler_source


def test_b1_main_window_constructs_and_passively_binds_page() -> None:
    source = (ROOT / "app/main_window.py").read_text(encoding="utf-8")
    assert "self._reconstruction_page = ReconstructionPage()" in source
    assert "self._mode_stack.addWidget(self._reconstruction_page)" in source
    assert "self._reconstruction_page.bind_controller(self._reconstruction_controller)" in source
