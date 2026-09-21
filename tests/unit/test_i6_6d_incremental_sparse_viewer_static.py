from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "app/widgets/reconstruction_page.py"
CANVAS = ROOT / "app/widgets/viewport/vispy_canvas.py"
CONTROLLER = ROOT / "app/reconstruction_controller.py"
BINDING = ROOT / "app/reconstruction_progressive_sfm_viewer_binding.py"


def _method_text(path: Path, class_name: str, method_name: str) -> str:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(
        node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )
    lines = text.splitlines()
    return "\n".join(lines[method.lineno - 1 : method.end_lineno])


def test_i6_6d_page_consumes_only_controller_validated_progressive_signal() -> None:
    page = PAGE.read_text(encoding="utf-8")
    binding = BINDING.read_text(encoding="utf-8")
    assert "bind_progressive_sfm_viewer(self, controller)" in page
    assert "controller.progressive_sparse_snapshot_changed.connect(" in binding
    assert "page.set_progressive_sparse_snapshot" in binding
    method = _method_text(PAGE, "ReconstructionPage", "set_progressive_sparse_snapshot")
    assert "build_progressive_sfm_viewer_frame(snapshot)" in method
    assert "show_progressive_sparse_reconstruction(" in method
    assert "start_offline" not in method
    assert "select_dense_provider" not in method
    assert "persistent_scene" not in method


def test_i6_6d_canvas_replaces_latest_revision_instead_of_accumulating_history() -> None:
    method = _method_text(CANVAS, "ReconstructionCanvas", "show_progressive_sparse_reconstruction")
    assert "update_sparse_landmarks(" in method
    assert "update_keyframe_poses(" in method
    assert "update_canvas=False" in method
    assert "_refresh_progressive_camera_path()" in method
    assert ".append(" not in method
    assert "np.concatenate" not in method
    assert "rev <= previous_revision" in method


def test_i6_6d_final_sparse_path_remains_the_existing_admitted_result_path() -> None:
    final_method = _method_text(CANVAS, "ReconstructionCanvas", "show_offline_sparse_reconstruction")
    progressive_method = _method_text(CANVAS, "ReconstructionCanvas", "show_progressive_sparse_reconstruction")
    clear_method = _method_text(CANVAS, "ReconstructionCanvas", "clear_progressive_sparse_reconstruction")
    assert "self.clear_live_state()" in final_method
    assert "_offline_visualization_loaded" not in progressive_method
    assert "persistent" not in progressive_method.lower()
    assert "if was_active and not self._offline_visualization_loaded" in clear_method


def test_i6_6d_does_not_modify_controller_transport_or_final_admission() -> None:
    controller = CONTROLLER.read_text(encoding="utf-8")
    assert "progressive_sparse_snapshot_changed = pyqtSignal(object)" in controller
    assert "_on_progressive_sparse_snapshot_received" in controller
    assert "_clear_progressive_sparse_snapshot()" in controller
    page = PAGE.read_text(encoding="utf-8")
    assert "Progressive sparse SfM • {label} • provisional" in page
