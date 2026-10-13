#!/usr/bin/env python3
"""Fixed-view preview for the I3.1 second-scene d12 transfer candidate."""
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
    p.add_argument("--i3-dir", type=Path, default=Path("/tmp/eia_i3_second_scene_campaign"))
    p.add_argument("--geometry", choices=("baseline", "candidate"), default="candidate")
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
    args = _args()
    directory = args.i3_dir.expanduser().resolve()
    summary = json.loads((directory / "i3_1_summary.json").read_text(encoding="utf-8"))
    if str(summary.get("status")) not in ("ready_for_manual_transfer_review", "degraded"):
        raise RuntimeError("I3.1 preview requires a completed transfer campaign.")
    artifact = Path(str(summary["artifacts"]["diagnostics_npz"])).expanduser().resolve()
    with np.load(artifact, allow_pickle=False) as p:
        points = np.asarray(p["points_reconstruction"], dtype=np.float64)
        rgb = np.asarray(p["colors_rgb"], dtype=np.uint8)
        reject = np.asarray(p["is_rejected_by_d12_transfer"], dtype=bool)
        core = np.asarray(p["is_core"], dtype=bool)
        far = np.asarray(p["is_far_peripheral"], dtype=bool)

    candidate = summary["candidate"]
    envelope = candidate["envelope"][args.geometry]
    lower = np.asarray(envelope["lower"], dtype=np.float64)
    upper = np.asarray(envelope["upper"], dtype=np.float64)

    if args.color_mode == "rgb":
        selected = np.ones(len(points), dtype=bool) if args.geometry == "baseline" else ~reject
        draw_points = points[selected]
        draw_colors = rgb[selected]
        legend = "baseline F Extended RGB" if args.geometry == "baseline" else "I3.1 d12 transfer candidate RGB"
    else:
        draw_points = points
        draw_colors = np.clip(np.rint(rgb.astype(np.float64) * 0.20), 0, 255).astype(np.uint8)
        draw_colors[reject] = [235, 70, 70]
        draw_colors[reject & far] = [245, 180, 50]
        draw_colors[reject & core] = [225, 65, 215]
        legend = "dim=kept | red=rejected | yellow=rejected far | magenta=rejected Core"

    draw_points, draw_colors = enforce_point_budget(draw_points, draw_colors, int(args.point_budget), seed=0)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow(); window.setWindowTitle("Depth & 3D Model Lab — I3.1 Second-Scene Transfer Review")
    central = QWidget(); layout = QVBoxLayout(central)
    ratio = float(candidate["envelope"][args.geometry]["full_to_robust_diagonal_ratio"])
    layout.addWidget(QLabel(
        f"I3.1 | {args.geometry} | {len(draw_points):,} rendered | {args.color_mode} | {args.view} | {args.fit} | "
        f"full/robust={ratio:.3f} | {legend} | counterfactual only"
    ))
    canvas = PointCloudCanvas(window); canvas.set_point_size(float(args.point_size)); canvas.set_grid_visible(True); canvas.update_points(draw_points, draw_colors)
    if args.fit == "full": canvas.fit_scene()
    else: _fit_robust(canvas, lower, upper)
    camera = getattr(canvas, "_camera", None)
    if camera is not None:
        camera.azimuth, camera.elevation = map(float, _VIEW_ANGLES[args.view])
        changed = getattr(camera, "view_changed", None)
        if callable(changed): changed()
    layout.addWidget(canvas.native, 1); window.setCentralWidget(central); window.resize(1366, 768); window.show()
    for _ in range(14): app.processEvents(); time.sleep(0.02)
    output = args.output.expanduser().resolve() if args.output else directory / f"i3_1_{args.geometry}_{args.color_mode}_{args.view}_{args.fit}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close(); raise RuntimeError(f"Unable to save I3.1 preview: {output}")
    print(f"I3.1 transfer preview: {output}")
    print(f"geometry={args.geometry} rendered_points={len(draw_points)} color_mode={args.color_mode} view={args.view} fit={args.fit} full/robust={ratio:.3f}")
    print("I3.1 counterfactual only: formal F/G and metric/measurement/mesh authority remain unchanged.")
    if args.show:
        code = app.exec(); canvas.close(); window.close(); return int(code)
    canvas.close(); window.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
