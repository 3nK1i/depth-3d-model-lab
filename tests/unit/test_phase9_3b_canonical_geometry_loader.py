from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from core.reconstruction import persistent_scene_from_pipeline_result
from tests.reconstruction_f5_helpers import run_f5_fixture

ROOT = Path(__file__).resolve().parents[2]


def _write_triangle(path: Path, *, colors: bool = True) -> None:
    suffix = " 1 0 0" if colors else ""
    lines = [
        f"v 0 0 0{suffix}",
        "v 1 0 0 0 1 0" if colors else "v 1 0 0",
        "v 0 1 0 0 0 1" if colors else "v 0 1 0",
        "vn 0 0 1",
        "vn 0 0 1",
        "vn 0 0 1",
        "f 1//1 2//2 3//3",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scene(tmp_path: Path, *, materialized: bool = True, counts: tuple[int, int] | None = (3, 1)):
    _fixture, _providers, result, snapshots = run_f5_fixture(
        tmp_path / "run", request_id="phase9-3b-loader"
    )
    snapshots = dict(snapshots)
    if materialized:
        surface = (tmp_path / "canonical_surface.obj").resolve()
        _write_triangle(surface)
        payload = {
            "surface_obj_path": str(surface),
            "metric_geometry_status": "arbitrary",
            "canonical_geometry_authoritative": True,
            "certified_metrology": False,
        }
        if counts is not None:
            payload["surface_vertex_count"] = counts[0]
            payload["surface_face_count"] = counts[1]
        snapshots["surface_mesh"] = payload
    return persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)


def test_3b_loads_integrity_bound_canonical_obj_into_viewer_coordinates(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    mesh = load_canonical_viewer_mesh(scene)
    assert mesh is not None
    assert mesh.scene_id == scene.scene_id
    assert mesh.artifact_id == scene.canonical_geometry_artifact_id
    assert mesh.vertex_count == 3
    assert mesh.face_count == 1
    np.testing.assert_allclose(
        mesh.vertices_viewer,
        np.asarray([[0, 0, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float32),
    )
    np.testing.assert_array_equal(mesh.faces, np.asarray([[0, 1, 2]], dtype=np.uint32))
    np.testing.assert_allclose(mesh.vertex_colors_rgba[:, 3], 1.0)
    assert mesh.certified_metrology is False
    assert mesh.source_sha256


def test_3b_scene_without_surface_reference_remains_valid_but_has_no_viewer_mesh(tmp_path: Path) -> None:
    scene = _scene(tmp_path, materialized=False)
    assert scene.canonical_geometry_authoritative is True
    assert load_canonical_viewer_mesh(scene) is None


def test_3b_rejects_surface_tamper_before_parsing(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    reference = next(
        item for item in scene.file_references
        if item.artifact_id == scene.canonical_geometry_artifact_id and item.role == "surface_obj_path"
    )
    Path(reference.path).write_text("v 9 9 9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="size changed|checksum changed"):
        load_canonical_viewer_mesh(scene)


def test_3b_rejects_persisted_count_disagreement(tmp_path: Path) -> None:
    scene = _scene(tmp_path, counts=(99, 1))
    with pytest.raises(ValueError, match="vertex count"):
        load_canonical_viewer_mesh(scene)


def test_3b_rejects_non_triangular_obj_without_reinterpreting_geometry(tmp_path: Path) -> None:
    _fixture, _providers, result, snapshots = run_f5_fixture(
        tmp_path / "quad-run", request_id="phase9-3b-quad"
    )
    surface = (tmp_path / "quad.obj").resolve()
    surface.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n",
        encoding="utf-8",
    )
    snapshots = dict(snapshots)
    snapshots["surface_mesh"] = {
        "surface_obj_path": str(surface),
        "metric_geometry_status": "arbitrary",
        "canonical_geometry_authoritative": True,
        "certified_metrology": False,
    }
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    with pytest.raises(ValueError, match="not a triangular face"):
        load_canonical_viewer_mesh(scene)


def test_3b_loader_does_not_import_heavy_backends_or_reconstruct_surface_result() -> None:
    path = ROOT / "app/reconstruction_canonical_geometry.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    for forbidden in (
        "pycolmap",
        "trimesh",
        "open3d",
        "gsplat",
        "nerfstudio",
        "depth_anything_3",
        "tinycudann",
    ):
        assert forbidden not in imported
    assert "from core.reconstruction.surface_mesh import" not in source
    assert "SurfaceMeshResult(" not in source
    assert "surface_obj_path" in source
    assert "certified_metrology=False" in source


def test_3b_real_vispy_canvas_contract_uses_one_mesh_visual_and_page_loader() -> None:
    canvas_source = (ROOT / "app/widgets/viewport/vispy_canvas.py").read_text(encoding="utf-8")
    page_source = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")

    assert "scene.visuals.Mesh(" in canvas_source
    assert "def set_canonical_mesh(" in canvas_source
    assert "vertex_colors=colors" in canvas_source
    assert "def clear_canonical_mesh(" in canvas_source
    assert "self._canonical_mesh_bounds_min" in canvas_source
    assert "self._canonical_mesh_bounds_max" in canvas_source
    assert "self.clear_canonical_mesh(update_canvas=False)" in canvas_source
    assert "load_canonical_viewer_mesh(scene)" in page_source
    assert "_refresh_canonical_geometry_viewer()" in page_source
    assert "SurfaceMeshResult(" not in page_source


def test_3b_real_vispy_clear_never_submits_zero_vertex_mesh() -> None:
    path = ROOT / "app/widgets/viewport/vispy_canvas.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    clear = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "clear_canonical_mesh"
    )

    # Real VisPy computes MeshData bounds inside MeshVisual.set_data().
    # Calling it with a zero-vertex array raises before the visual can be
    # hidden.  Clear must therefore be scenegraph/visibility based only.
    set_data_calls = [
        node
        for node in ast.walk(clear)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_data"
    ]
    assert set_data_calls == []

    segment = ast.get_source_segment(source, clear) or ""
    assert "self._canonical_mesh.visible = False" in segment
    assert "self._canonical_mesh.parent = None" in segment
    assert "scene.visuals.Mesh(" in segment
