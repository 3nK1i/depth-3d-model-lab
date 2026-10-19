#!/usr/bin/env python3
"""Fixed-view preview for the promoted I3.2 second-scene Dense RGB geometry."""
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

_VIEW_ANGLES = {"oblique": (32.0, 24.0), "front": (0.0, 0.0), "side": (90.0, 0.0), "top": (0.0, 89.0)}


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i3-2-dir", type=Path, default=Path("/tmp/depth3d_i3_2_second_scene_promotion"))
    p.add_argument("--color-mode", choices=("rgb", "decision"), default="rgb")
    p.add_argument("--view", choices=tuple(_VIEW_ANGLES), default="oblique")
    p.add_argument("--fit", choices=("full", "robust"), default="full")
    p.add_argument("--point-budget", type=int, default=250_000)
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def _fit_robust(canvas: PointCloudCanvas, lower: np.ndarray, upper: np.ndarray) -> None:
    camera = getattr(canvas, "_camera", None)
    if camera is None:
        return
    center = (lower + upper) * 0.5
    diagonal = max(float(np.linalg.norm(upper - lower)), 0.5)
    camera.center = tuple(float(v) for v in center)
    camera.distance = max(diagonal * 1.35, 1.5)
    changed = getattr(camera, "view_changed", None)
    if callable(changed):
        changed()


def main() -> int:
    a = _args()
    directory = a.i3_2_dir.expanduser().resolve()
    summary = json.loads((directory / "i3_2_summary.json").read_text(encoding="utf-8"))
    if not bool(summary.get("automated_pass")):
        raise RuntimeError("I3.2 preview requires automated_pass=True.")
    i31_dir = Path(str(summary["source_i3_1_dir"])).expanduser().resolve()
    i31 = json.loads((i31_dir / "i3_1_summary.json").read_text(encoding="utf-8"))
    diag_path = Path(str(i31["artifacts"]["diagnostics_npz"])).expanduser().resolve()
    with np.load(diag_path, allow_pickle=False) as p:
        baseline_points = np.asarray(p["points_reconstruction"], dtype=np.float64)
        baseline_rgb = np.asarray(p["colors_rgb"], dtype=np.uint8)
        reject = np.asarray(p["is_rejected_by_d12_transfer"], dtype=bool)
        core = np.asarray(p["is_core"], dtype=bool)
        far = np.asarray(p["is_far_peripheral"], dtype=bool)

    env = summary["requalification"]["envelope"]["candidate"]
    lower = np.asarray(env["lower"], dtype=np.float64)
    upper = np.asarray(env["upper"], dtype=np.float64)
    ratio = float(env["full_to_robust_diagonal_ratio"])

    if a.color_mode == "rgb":
        points = baseline_points[~reject]
        colors = baseline_rgb[~reject]
        legend = "promoted second-scene Dense RGB"
    else:
        points = baseline_points
        colors = np.clip(np.rint(baseline_rgb.astype(np.float64) * 0.20), 0, 255).astype(np.uint8)
        colors[reject] = [235, 70, 70]
        colors[reject & far] = [245, 180, 50]
        colors[reject & core] = [225, 65, 215]
        legend = "dim=kept | red=rejected | yellow=rejected far | magenta=rejected Core"

    points, colors = enforce_point_budget(points, colors, int(a.point_budget), seed=0)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow(); window.setWindowTitle("Depth & 3D Model Lab — I3.2 Promoted Second-Scene Review")
    central = QWidget(); layout = QVBoxLayout(central)
    layout.addWidget(QLabel(
        f"I3.2 | promoted | {len(points):,} rendered | {a.color_mode} | {a.view} | {a.fit} | "
        f"full/robust={ratio:.3f} | {legend} | final manual review"
    ))
    canvas = PointCloudCanvas(window); canvas.set_point_size(float(a.point_size)); canvas.set_grid_visible(True); canvas.update_points(points, colors)
    if a.fit == "full":
        canvas.fit_scene()
    else:
        _fit_robust(canvas, lower, upper)
    camera = getattr(canvas, "_camera", None)
    if camera is not None:
        camera.azimuth, camera.elevation = map(float, _VIEW_ANGLES[a.view])
        changed = getattr(camera, "view_changed", None)
        if callable(changed): changed()
    layout.addWidget(canvas.native, 1); window.setCentralWidget(central); window.resize(1366, 768); window.show()
    for _ in range(14): app.processEvents(); time.sleep(0.02)
    output = a.output.expanduser().resolve() if a.output else directory / f"i3_2_promoted_{a.color_mode}_{a.view}_{a.fit}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close(); raise RuntimeError(f"Unable to save I3.2 preview: {output}")
    print(f"I3.2 promoted preview: {output}")
    print(f"rendered_points={len(points)} color_mode={a.color_mode} view={a.view} fit={a.fit} full/robust={ratio:.3f}")
    print("I3.2 review only: promoted F/G remain arbitrary-scale; metric/measurement/mesh authority remain unpromoted.")
    if a.show:
        code = app.exec(); canvas.close(); window.close(); return int(code)
    canvas.close(); window.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
