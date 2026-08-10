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


def test_b4_status_contract_survives_b5_navigation_enablement() -> None:
    tabs = (ROOT / "app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    enabled = tabs.split("_ENABLED_MODES =", 1)[1].split("}", 1)[0]
    assert "AppMode.RECONSTRUCTION" not in enabled
    handler = _method_source(ROOT / "app/main_window.py", "MainWindow", "_on_mode_changed")
    assert "AppMode.RECONSTRUCTION" in handler


def test_b4_page_consumes_existing_events_without_raw_log_binding() -> None:
    source = _method_source(ROOT / "app/widgets/reconstruction_page.py", "ReconstructionPage", "bind_controller")
    assert "offline_event_received.connect(self.set_offline_event)" in source
    assert "live_observation_ready.connect(self.set_live_observation)" in source
    assert "offline_result_ready.connect(self._on_offline_result_ready)" in source
    assert "offline_log_received.connect" not in source


def test_b4_progress_is_presentation_only_and_does_not_start_backends() -> None:
    source = (ROOT / "app/reconstruction_status.py").read_text(encoding="utf-8")
    assert "start_live" not in source
    assert "start_offline" not in source
    assert "QProcess" not in source


def test_b4_status_module_has_no_qt_or_heavy_backend_imports() -> None:
    source = (ROOT / "app/reconstruction_status.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    forbidden = ("PyQt6", "pycolmap", "gsplat", "nerfstudio", "depth_anything_3", "stella_vslam")
    assert not any(name.startswith(forbidden) for name in imports)


def test_b4_live_progress_is_indeterminate_instead_of_fabricated_percentage() -> None:
    source = (ROOT / "app/reconstruction_status.py").read_text(encoding="utf-8")
    assert '"Live tracking"' in source
    assert "indeterminate" in source
    assert "progress_fraction" in source


def test_b4_offline_stages_reflect_current_real_a3_worker_scope() -> None:
    source = (ROOT / "app/reconstruction_status.py").read_text(encoding="utf-8")
    assert '"source_ingestion": "Ingesting source"' in source
    assert '"camera_pose": "Camera reconstruction"' in source
    assert "Dense geometry" not in source
    assert "NeRF" not in source
