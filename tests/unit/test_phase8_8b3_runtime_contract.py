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


def test_b3_navigation_remains_disabled_and_unrouted() -> None:
    tabs = (ROOT / "app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    enabled = tabs.split("_ENABLED_MODES =", 1)[1].split("}", 1)[0]
    assert "AppMode.RECONSTRUCTION" not in enabled
    handler = _method_source(ROOT / "app/main_window.py", "MainWindow", "_on_mode_changed")
    assert "AppMode.RECONSTRUCTION" not in handler


def test_b3_source_selection_handlers_still_do_not_start_backends() -> None:
    path = ROOT / "app/widgets/reconstruction_page.py"
    for method in ("_select_live_source", "_choose_image_sequence", "_choose_recorded_video"):
        source = _method_source(path, "ReconstructionPage", method)
        assert "start_selected_" not in source
        assert "start_live(" not in source
        assert "start_offline(" not in source


def test_b3_runtime_controls_delegate_to_hardened_controller_only() -> None:
    source = _method_source(
        ROOT / "app/widgets/reconstruction_page.py", "ReconstructionPage", "_start_runtime"
    )
    assert "start_selected_live" in source
    assert "start_selected_offline" in source
    assert "ExternalStella" not in source
    assert "pycolmap" not in source.casefold()


def test_b3_production_ui_and_controller_have_no_heavy_backend_imports() -> None:
    forbidden = ("pycolmap", "gsplat", "nerfstudio", "tinycudann", "depth_anything_3", "stella_vslam")
    for relative in ("app/reconstruction_controller.py", "app/widgets/reconstruction_page.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(name.startswith(forbidden) for name in imports)


def test_b3_live_start_offers_calibrated_or_explicit_non_metric_test_mode() -> None:
    page = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    controller = (ROOT / "app/reconstruction_controller.py").read_text(encoding="utf-8")
    live_bridge = (ROOT / "app/live_slam_controller.py").read_text(encoding="utf-8")
    assert "Use calibration file" in page
    assert "Run Test Mode" in page
    assert "TEST MODE • NON-METRIC" in page
    assert "load_live_camera_calibration" in page
    assert "build_estimated_live_camera" in page
    assert "estimate_camera_geometry_from_horizontal_fov" in controller
    assert "CameraIntrinsicsPolicy.ALLOW_ESTIMATED" in page
    assert "camera_intrinsics_policy=camera_intrinsics_policy" in live_bridge
