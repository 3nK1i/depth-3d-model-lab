from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _method_source(path: Path, class_name: str, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return ast.get_source_segment(source, method) or ""


def test_c1_visualizer_is_presentation_only_and_dependency_light() -> None:
    helper = (ROOT / "app/reconstruction_visualization.py").read_text(encoding="utf-8")
    page = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    combined_tree = ast.parse(helper + "\n" + page)
    imported_roots: set[str] = set()
    for node in ast.walk(combined_tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    for forbidden in ("pycolmap", "gsplat", "nerfstudio", "depth_anything_3", "tinycudann"):
        assert forbidden not in imported_roots
    assert "T_reconstruction_from_camera" in helper
    assert "ARBITRARY GAUGE" in page


def test_c1_page_coalesces_high_rate_pose_updates_before_gl_rendering() -> None:
    source = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    assert "self._visual_pose_timer.setInterval(33)" in source
    observation = _method_source(
        ROOT / "app/widgets/reconstruction_page.py", "ReconstructionPage", "set_live_observation"
    )
    assert "self._pending_visual_pose" in observation
    assert "self._reconstruction_canvas.update_live_pose" not in observation
    flush = _method_source(
        ROOT / "app/widgets/reconstruction_page.py", "ReconstructionPage", "_flush_pending_visual_pose"
    )
    assert "self._reconstruction_canvas.update_live_pose(matrix)" in flush


def test_c1_trajectory_buffer_contract_remains_bounded_after_c2_promotion() -> None:
    canvas = (ROOT / "app/widgets/viewport/vispy_canvas.py").read_text(encoding="utf-8")
    assert "MAX_TRAJECTORY_POSES = 20_000" in canvas
    assert "self._trajectory_positions" in canvas
    assert "def set_trajectory_visible" in canvas


def test_c1_navigation_and_backend_execution_contracts_remain_unchanged() -> None:
    tabs = (ROOT / "app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    assert "_RECONSTRUCTION_NAVIGATION_ENABLED = True" in tabs
    page = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    # Existing B3 Start handlers remain the only backend-start boundary.
    assert "start_selected_live" in page
    assert "start_selected_offline" in page
    canvas = (ROOT / "app/widgets/viewport/vispy_canvas.py").read_text(encoding="utf-8")
    for forbidden in ("start_selected_live", "start_selected_offline", "pycolmap", "Stella"):
        assert forbidden not in canvas


def test_c1_shutdown_stops_producers_before_releasing_gl_canvas() -> None:
    close_handler = _method_source(ROOT / "app/main_window.py", "MainWindow", "closeEvent")
    producer = close_handler.index("self._reconstruction_controller.shutdown()")
    surface = close_handler.index("self._reconstruction_page.shutdown_viewport()")
    assert producer < surface
    page_shutdown = _method_source(
        ROOT / "app/widgets/reconstruction_page.py", "ReconstructionPage", "shutdown_viewport"
    )
    assert "self._viewport_shutdown = True" in page_shutdown
    assert "self._visual_pose_timer.stop()" in page_shutdown
