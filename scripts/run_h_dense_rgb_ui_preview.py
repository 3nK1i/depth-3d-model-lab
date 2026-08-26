#!/usr/bin/env python3
"""Manual Phase-H Dense RGB Reconstruction workspace preview.

Loads only the self-contained Phase-G bundle through the production H UI bridge.
No reconstruction backend is launched.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.widgets.reconstruction_page import ReconstructionPage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", required=True, help="Phase-G persistent Dense RGB scene directory or manifest.")
    parser.add_argument("--representation", choices=("extended", "core"), default="extended")
    parser.add_argument("--width", type=int, default=1366)
    parser.add_argument("--height", type=int, default=860)
    parser.add_argument("--show", action="store_true", help="Keep the interactive Reconstruction workspace open.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    page = ReconstructionPage()
    page.resize(max(1024, int(args.width)), max(720, int(args.height)))
    page.set_persistent_dense_rgb_scene(
        args.scene_dir,
        representation=args.representation,
        fit_scene=True,
    )
    presentation = page.dense_rgb_presentation
    if presentation is None:
        raise RuntimeError("Phase H did not bind a Dense RGB presentation.")
    canvas = page._reconstruction_canvas
    print("H Dense RGB Reconstruction UI integration: PASS")
    print(f"scene_id={presentation.scene_id}")
    print(f"representation={presentation.representation} points={presentation.point_count}")
    print(f"views={presentation.source_view_count} voxel={presentation.voxel_size_reconstruction_units:.6f}")
    print(
        "canonical_geometry_authoritative=True metric_scale_claimed=False "
        "measurement_authoritative=False surface_mesh_generated=False"
    )
    print(
        f"viewport_loaded={canvas.dense_rgb_points_loaded} "
        f"display_points={canvas.dense_rgb_display_count} backend_started=False"
    )
    print("next=I - Offline Dense Reconstruction Quality Campaign (after H manual UI validation)")
    if not args.show:
        page.shutdown_viewport()
        page.deleteLater()
        app.processEvents()
        return 0
    page.show()
    page.raise_()
    page.activateWindow()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
