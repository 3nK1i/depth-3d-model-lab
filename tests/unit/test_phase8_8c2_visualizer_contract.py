from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _method_source(path: Path, class_name: str, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    return ast.get_source_segment(source, method) or ""


def test_c2_promotes_bounded_trajectory_to_visible_decimated_rendering() -> None:
    canvas = (ROOT / "app/widgets/viewport/vispy_canvas.py").read_text(encoding="utf-8")
    assert "MAX_TRAJECTORY_POSES = 20_000" in canvas
    assert "MAX_TRAJECTORY_DISPLAY_POINTS = 4_096" in canvas
    assert "self._trajectory_visible = True" in canvas
    assert "trajectory_display_indices" in canvas
    assert "self._trajectory_start" in canvas


def test_c2_replaces_dominant_xyzaxis_with_restrained_world_origin_axes() -> None:
    canvas = (ROOT / "app/widgets/viewport/vispy_canvas.py").read_text(encoding="utf-8")
    reconstruction_class = canvas.split("class ReconstructionCanvas", 1)[1]
    assert "scene.visuals.XYZAxis" not in reconstruction_class
    assert "adaptive_world_axis_length" in reconstruction_class
    assert "self._axis_x" in reconstruction_class
    assert "self._axis_y" in reconstruction_class
    assert "self._axis_z" in reconstruction_class


def test_c2_adds_fixed_orientation_gizmo_without_geometry_authority_changes() -> None:
    page = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    gizmo = (ROOT / "app/widgets/reconstruction_orientation_gizmo.py").read_text(encoding="utf-8")
    helper = (ROOT / "app/reconstruction_visualization.py").read_text(encoding="utf-8")
    assert "ReconstructionOrientationGizmo" in page
    assert "AlignRight | Qt.AlignmentFlag.AlignBottom" in page
    assert "WA_TransparentForMouseEvents" in gizmo
    assert "orientation_gizmo_projection" in helper
    assert "T_reconstruction_from_camera" in helper


def test_c2_trajectory_toggle_is_presentation_only() -> None:
    page = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    canvas = (ROOT / "app/widgets/viewport/vispy_canvas.py").read_text(encoding="utf-8")
    assert 'tooltip="Show or hide live camera trajectory"' in page
    assert "set_trajectory_visible" in page
    method = _method_source(
        ROOT / "app/widgets/viewport/vispy_canvas.py", "ReconstructionCanvas", "set_trajectory_visible"
    )
    for forbidden in ("start_selected_live", "start_selected_offline", "pycolmap", "Stella"):
        assert forbidden not in method


def test_c2_remains_dependency_light_and_does_not_touch_reconstruction_core() -> None:
    sources = [
        ROOT / "app/reconstruction_visualization.py",
        ROOT / "app/widgets/reconstruction_orientation_gizmo.py",
        ROOT / "app/widgets/reconstruction_page.py",
        ROOT / "app/widgets/viewport/vispy_canvas.py",
    ]
    imported_roots: set[str] = set()
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
    for forbidden in ("pycolmap", "gsplat", "nerfstudio", "depth_anything_3", "tinycudann"):
        assert forbidden not in imported_roots
