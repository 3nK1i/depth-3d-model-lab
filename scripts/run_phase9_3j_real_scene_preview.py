#!/usr/bin/env python3
"""Real Qt/VisPy preview of the Phase 9.3J qualified persistent canonical scene.

This script loads an already-materialized scene only. It never launches DA3,
reconstruction, Gaussian, or NeRF backends.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Mapping

os.environ.pop("DEPTH3D_TEST_FAKE_VISPY", None)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from core.reconstruction.persistent_scene import read_persistent_reconstruction_scene


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=False)
    source.add_argument("--scene-manifest", type=Path, default=None)
    source.add_argument(
        "--qualification-summary",
        type=Path,
        default=Path("/tmp/phase9_3j_real_scene/phase9_3j_qualification_summary.json"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/phase9_3j_real_scene_preview.png"),
    )
    p.add_argument("--show", action="store_true")
    return p


def _read_json(path: Path) -> Mapping[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise RuntimeError("Qualification summary root must be an object.")
    return raw


def _resolve_scene(args: argparse.Namespace) -> Path:
    if args.scene_manifest is not None:
        target = args.scene_manifest.expanduser().resolve()
    else:
        summary = _read_json(args.qualification_summary.expanduser().resolve())
        persistent = summary.get("persistent_scene")
        if not isinstance(persistent, Mapping):
            raise RuntimeError("Qualification summary does not contain persistent_scene evidence.")
        target = Path(str(persistent.get("manifest", ""))).expanduser().resolve()
    if not target.is_file():
        raise RuntimeError(f"Persistent scene manifest does not exist: {target}")
    return target


def _pump(app: QApplication, count: int = 16) -> None:
    for _ in range(count):
        app.processEvents()
        time.sleep(0.02)


def main() -> int:
    args = _parser().parse_args()
    scene_manifest = _resolve_scene(args)
    scene = read_persistent_reconstruction_scene(scene_manifest)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = MainWindow()
    window.resize(1366, 768)
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    controller = window._reconstruction_controller
    page = window._reconstruction_page
    loaded = controller.load_persistent_scene(scene_manifest)
    if loaded.scene_id != scene.scene_id:
        raise RuntimeError("Application reload changed the qualified persistent-scene identity.")
    window.show()
    _pump(app, 20)

    canvas = page._reconstruction_canvas
    if not canvas.canonical_mesh_loaded:
        raise RuntimeError("9.3J real preview did not materialize the canonical surface.")
    if page.canonical_camera_presentation is None or page.canonical_camera_presentation.persisted_pose_count < 2:
        raise RuntimeError("9.3J real preview did not load persisted camera poses.")
    if not scene.canonical_geometry_authoritative or scene.certified_metrology:
        raise RuntimeError("9.3J real preview changed canonical/metrology authority.")
    if page.canonical_semantic_presentation is not None:
        raise RuntimeError("Real DA3 scene unexpectedly fabricated canonical semantics.")
    if page._semantic_object_combo.isEnabled():
        raise RuntimeError("Real DA3 scene unexpectedly enabled semantic object selection.")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        raise RuntimeError(f"Unable to save 9.3J preview screenshot: {output}")

    print(f"Phase 9.3J real persistent-scene preview: {output}")
    print(f"scene_id={scene.scene_id} scene_state={scene.state.value}")
    print(
        f"canonical_vertices={canvas.canonical_mesh_vertex_count} "
        f"canonical_faces={canvas.canonical_mesh_face_count} "
        f"cameras={page.canonical_camera_presentation.persisted_pose_count}"
    )
    print("Canonical geometry remains authoritative; metric scale/certified metrology were not invented.")
    print("Semantic/inspection context remains unavailable when the real DA3 scene does not contain it.")
    print("No DA3/reconstruction/Gaussian/NeRF backend was launched by persistent-scene reload or preview.")

    if args.show:
        code = app.exec()
        window.close()
        print("Phase 9.3J real-scene viewer/controller resources released on shutdown.")
        return int(code)
    window.close()
    print("Phase 9.3J real-scene viewer/controller resources released on shutdown.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
