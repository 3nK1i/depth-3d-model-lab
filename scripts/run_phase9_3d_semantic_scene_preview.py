#!/usr/bin/env python3
"""Real-VisPy preview for Phase 9.3D semantic objects in canonical 3D."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
import time

# Exercise the actual VisPy/OpenGL ReconstructionCanvas, not the Qt test double.
os.environ.pop("DEPTH3D_TEST_FAKE_VISPY", None)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.widgets.reconstruction_page import ReconstructionPage
from tests.phase9_3d_helpers import make_phase9_3d_scene


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/phase9_3d_semantic_scene_preview.png"),
        help="PNG screenshot path",
    )
    parser.add_argument("--show", action="store_true", help="Keep the real VisPy preview open.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    with tempfile.TemporaryDirectory(prefix="phase9_3d_preview_") as tmp:
        root = Path(tmp).resolve()
        scene, semantic, _surface = make_phase9_3d_scene(
            root / "source", request_id="phase9-3d-preview"
        )
        page = ReconstructionPage()
        page.resize(1366, 768)
        page.set_persistent_scene(scene, "phase9_3d_preview/persistent_reconstruction_scene.json")
        page._canonical_scene_group.set_expanded(True)
        page._semantic_objects_group.set_expanded(True)
        page.show()
        for _ in range(15):
            app.processEvents()
            time.sleep(0.02)

        canvas = page._reconstruction_canvas
        presentation = page.canonical_semantic_presentation
        if page.canonical_viewer_mesh is None or not canvas.canonical_mesh_loaded:
            raise RuntimeError("Phase 9.3D preview did not load the persistent canonical surface.")
        if presentation is None or not canvas.canonical_semantic_loaded:
            raise RuntimeError("Phase 9.3D preview did not bind persistent semantic objects.")
        if presentation.surface_highlight_object_count < 1:
            raise RuntimeError("Phase 9.3D preview has no validated semantic surface membership.")

        # Select the first real persisted semantic object so the screenshot and
        # interactive preview exercise the selected-surface overlay and label.
        page._semantic_object_combo.setCurrentIndex(1)
        for _ in range(8):
            app.processEvents()
            time.sleep(0.02)
        if not canvas.selected_canonical_semantic_surface_visible:
            raise RuntimeError("Selected semantic canonical-surface highlight is not visible.")
        if not canvas.selected_canonical_semantic_label_visible:
            raise RuntimeError("Selected semantic label is not visible.")

        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if not page.grab().save(str(output), "PNG"):
            raise RuntimeError(f"Unable to save preview screenshot: {output}")
        print(f"Phase 9.3D real-VisPy semantic scene preview: {output}")
        print(
            f"canonical_vertices={page.canonical_viewer_mesh.vertex_count} "
            f"canonical_faces={page.canonical_viewer_mesh.face_count} "
            f"semantic_objects={len(semantic.objects)} "
            f"surface_bound={presentation.surface_bound_object_count} "
            f"highlightable={presentation.surface_highlight_object_count}"
        )
        print(
            "Semantic authority=semantic_context; canonical geometry authority unchanged; "
            "certified_metrology=False."
        )
        print("No YOLO/SAM2/reconstruction/DA3/Gaussian/NeRF backend was launched by scene loading.")

        if args.show:
            exit_code = app.exec()
            page.shutdown_viewport()
            if canvas.canonical_mesh_loaded or canvas.canonical_semantic_loaded:
                raise RuntimeError("Canonical semantic viewer resources remained loaded after shutdown.")
            page.close()
            print("Canonical mesh/semantic viewer resources released on viewport shutdown.")
            return int(exit_code)

        page.shutdown_viewport()
        if canvas.canonical_mesh_loaded or canvas.canonical_semantic_loaded:
            raise RuntimeError("Canonical semantic viewer resources remained loaded after shutdown.")
        page.close()
        print("Canonical mesh/semantic viewer resources released on viewport shutdown.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
