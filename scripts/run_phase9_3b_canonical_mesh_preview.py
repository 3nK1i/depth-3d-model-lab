#!/usr/bin/env python3
"""Real-VisPy UI preview for Phase 9.3B persistent canonical surface rendering."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
import time

# 9.3B must validate the actual VisPy mesh path, not the pytest Qt-only double.
os.environ.pop("DEPTH3D_TEST_FAKE_VISPY", None)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.widgets.reconstruction_page import ReconstructionPage
from core.reconstruction import persistent_scene_from_pipeline_result
from tests.reconstruction_f5_helpers import run_f5_fixture


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/phase9_3b_canonical_mesh_preview.png"),
        help="PNG screenshot path",
    )
    parser.add_argument("--show", action="store_true", help="Keep the real VisPy preview open.")
    return parser


def _write_colored_box(path: Path) -> None:
    vertices = (
        (-1.6, -1.0, -0.7, 0.88, 0.22, 0.28),
        (1.6, -1.0, -0.7, 0.22, 0.76, 0.42),
        (1.6, 1.0, -0.7, 0.22, 0.48, 0.92),
        (-1.6, 1.0, -0.7, 0.92, 0.72, 0.20),
        (-1.6, -1.0, 0.7, 0.78, 0.28, 0.88),
        (1.6, -1.0, 0.7, 0.20, 0.82, 0.88),
        (1.6, 1.0, 0.7, 0.94, 0.94, 0.94),
        (-1.6, 1.0, 0.7, 0.36, 0.56, 0.96),
    )
    faces = (
        (1, 2, 3), (1, 3, 4),
        (5, 7, 6), (5, 8, 7),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 8), (3, 8, 4),
        (4, 8, 5), (4, 5, 1),
    )
    lines = ["# Phase 9.3B synthetic persistent canonical surface"]
    lines.extend("v " + " ".join(str(value) for value in vertex) for vertex in vertices)
    lines.extend("f " + " ".join(str(value) for value in face) for face in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parser().parse_args()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    with tempfile.TemporaryDirectory(prefix="phase9_3b_preview_") as tmp:
        root = Path(tmp).resolve()
        _fixture, _providers, result, snapshots = run_f5_fixture(
            root / "run", request_id="phase9-3b-preview"
        )
        surface = (root / "canonical_surface.obj").resolve()
        _write_colored_box(surface)
        snapshots = dict(snapshots)
        snapshots["surface_mesh"] = {
            "surface_obj_path": str(surface),
            "surface_vertex_count": 8,
            "surface_face_count": 12,
            "metric_geometry_status": "arbitrary",
            "canonical_geometry_authoritative": True,
            "certified_metrology": False,
        }
        scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)

        page = ReconstructionPage()
        page.resize(1366, 768)
        page.set_persistent_scene(scene, "phase9_3b_preview/persistent_reconstruction_scene.json")
        page._canonical_scene_group.set_expanded(True)
        page.show()
        for _ in range(12):
            app.processEvents()
            time.sleep(0.02)

        if page.canonical_viewer_mesh is None or not page._reconstruction_canvas.canonical_mesh_loaded:
            raise RuntimeError("Phase 9.3B preview did not load the persistent canonical surface.")

        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if not page.grab().save(str(output), "PNG"):
            raise RuntimeError(f"Unable to save preview screenshot: {output}")
        print(f"Phase 9.3B real-VisPy canonical mesh preview: {output}")
        print("canonical_vertices=8 canonical_faces=12 canonical_authoritative=True certified_metrology=False")
        print("No reconstruction/DA3/Gaussian/NeRF backend was launched.")

        if args.show:
            exit_code = app.exec()
            page.shutdown_viewport()
            if page._reconstruction_canvas.canonical_mesh_loaded:
                raise RuntimeError("Canonical viewer resource remained loaded after viewport shutdown.")
            page.close()
            print("Canonical viewer resource released on viewport shutdown.")
            return int(exit_code)
        page.shutdown_viewport()
        if page._reconstruction_canvas.canonical_mesh_loaded:
            raise RuntimeError("Canonical viewer resource remained loaded after viewport shutdown.")
        page.close()
        print("Canonical viewer resource released on viewport shutdown.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
