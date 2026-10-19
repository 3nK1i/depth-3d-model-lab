#!/usr/bin/env python3
"""Fixed-view preview of the I2.6 promoted persistent Dense RGB scene."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.pop("DEPTH3D_TEST_FAKE_VISPY", None)

from app.widgets.viewport.vispy_canvas import PointCloudCanvas  # noqa: E402
from core.geometry.sampling import enforce_point_budget  # noqa: E402
from core.reconstruction.dense_scene_persistence import restore_persistent_dense_geometry  # noqa: E402

_VIEW_ANGLES = {"oblique": (32.0, 24.0), "front": (0.0, 0.0), "side": (90.0, 0.0), "top": (0.0, 89.0)}


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i2-6-dir", type=Path, default=Path("/tmp/depth3d_i2_6_manifold_filter_promotion"))
    p.add_argument("--representation", choices=("extended", "core"), default="extended")
    p.add_argument("--view", choices=tuple(_VIEW_ANGLES), default="oblique")
    p.add_argument("--point-budget", type=int, default=250_000)
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _args()
    directory = args.i2_6_dir.expanduser().resolve()
    summary = json.loads((directory / "i2_6_summary.json").read_text(encoding="utf-8"))
    if str(summary.get("status")) != "pass":
        raise RuntimeError("I2.6 promoted-scene preview requires status=pass.")
    manifest = Path(str(summary["artifacts"]["persistent_scene_manifest"])).expanduser().resolve()
    geometry = restore_persistent_dense_geometry(manifest, representation=args.representation)
    points = np.asarray(geometry.points_reconstruction, dtype=np.float64)
    colors = np.asarray(geometry.colors_rgb, dtype=np.uint8)
    points, colors = enforce_point_budget(points, colors, int(args.point_budget), seed=0)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow(); window.setWindowTitle("Depth & 3D Model Lab — I2.6 Promoted Dense RGB Scene")
    central = QWidget(); layout = QVBoxLayout(central)
    comparison = summary["promotion"]["qualified_counterfactual_comparison"]
    layout.addWidget(QLabel(
        f"I2.6 | {args.representation} | {len(points):,} rendered | {args.view} | "
        f"full/robust={float(comparison['candidate_full_to_robust_ratio']):.3f} | "
        "production F/G promoted | arbitrary SfM scale | no mesh authority"
    ))
    canvas = PointCloudCanvas(window)
    canvas.set_point_size(float(args.point_size)); canvas.set_grid_visible(True); canvas.update_points(points, colors); canvas.fit_scene()
    camera = getattr(canvas, "_camera", None)
    if camera is not None:
        camera.azimuth, camera.elevation = map(float, _VIEW_ANGLES[args.view])
        changed = getattr(camera, "view_changed", None)
        if callable(changed): changed()
    layout.addWidget(canvas.native, 1); window.setCentralWidget(central); window.resize(1366, 768); window.show()
    for _ in range(14): app.processEvents(); time.sleep(.02)
    output = args.output.expanduser().resolve() if args.output else directory / f"i2_6_{args.representation}_{args.view}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close(); raise RuntimeError(f"Unable to save I2.6 preview: {output}")
    print(f"I2.6 promoted scene preview: {output}")
    print(f"representation={args.representation} rendered_points={len(points)} view={args.view} scene_id={summary['scene_id']}")
    print("I2.6 production F/G promoted; metric/measurement/mesh authority remains unpromoted.")
    if args.show:
        code = app.exec(); canvas.close(); window.close(); return int(code)
    canvas.close(); window.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
