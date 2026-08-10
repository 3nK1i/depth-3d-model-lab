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


def test_b5_enables_reconstruction_through_a_separate_additive_gate() -> None:
    source = (ROOT / "app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    legacy = source.split("_ENABLED_MODES =", 1)[1].split("}", 1)[0]
    assert "AppMode.RECONSTRUCTION" not in legacy
    for mode_name in (
        "AppMode.LIVE_3D",
        "AppMode.INSPECTION",
        "AppMode.STABLE_VIDEO",
        "AppMode.SNAPSHOT",
        "AppMode.BENCHMARK",
    ):
        assert mode_name in legacy
    assert "_RECONSTRUCTION_NAVIGATION_ENABLED = True" in source
    assert "_mode_is_enabled(mode)" in source


def test_b5_routes_reconstruction_to_the_existing_page_only() -> None:
    handler = _method_source(ROOT / "app/main_window.py", "MainWindow", "_on_mode_changed")
    assert "if mode is AppMode.RECONSTRUCTION" in handler
    assert "self._mode_stack.setCurrentWidget(self._reconstruction_page)" in handler
    for forbidden in (
        "start_live(",
        "start_selected_live(",
        "start_offline(",
        "start_selected_offline(",
        "ExternalStella",
        "pycolmap",
    ):
        assert forbidden not in handler


def test_b5_navigation_activation_does_not_change_reconstruction_core_or_page_start_contract() -> None:
    mode_source = (ROOT / "app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    page_source = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    assert "core.reconstruction" not in mode_source
    assert "pycolmap" not in mode_source.casefold()
    assert "gsplat" not in mode_source.casefold()
    assert "nerfstudio" not in mode_source.casefold()
    assert "start_selected_live" in page_source
    assert "start_selected_offline" in page_source


def test_b5_main_window_tracks_reconstruction_activity_for_navigation_locking() -> None:
    source = (ROOT / "app/main_window.py").read_text(encoding="utf-8")
    assert "live_state_changed.connect" in source
    assert "offline_state_changed.connect" in source
    assert "offline_run_finished.connect" in source
    handler = _method_source(
        ROOT / "app/main_window.py", "MainWindow", "_on_reconstruction_activity_changed"
    )
    assert "self._reconstruction_controller.is_active" in handler
    assert "self._mode_tabs.set_navigation_locked(locked)" in handler


def test_b5_navigation_lock_considers_existing_long_running_job_owners() -> None:
    handler = _method_source(
        ROOT / "app/main_window.py", "MainWindow", "_on_reconstruction_activity_changed"
    )
    for owner in (
        "self._benchmark_controller.is_running",
        "self._snapshot_controller.is_running",
        "self._stable_video_controller.is_active",
        "self._inspection_controller.is_running",
        "self._inspection_export_controller.is_running",
    ):
        assert owner in handler
