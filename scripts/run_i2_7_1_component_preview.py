#!/usr/bin/env python3
"""Visualize I2.7.1 baseline→promoted component lineage."""
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
    p.add_argument("--i2-7-1-dir", type=Path, default=Path("/tmp/eia_i2_7_1_detached_component_qualification"))
    p.add_argument("--color-mode", choices=("lineage", "rgb"), default="lineage")
    p.add_argument("--view", choices=tuple(_VIEW_ANGLES), default="oblique")
    p.add_argument("--point-budget", type=int, default=250_000)
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def _lineage_colors(codes: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    # 0 promoted dominant, 1 surviving pre-existing detached, 2 newly detached,
    # 3 newly detached Core, 4 d12-removed voxel.
    colors = np.clip(np.rint(rgb.astype(np.float64) * 0.30), 0, 255).astype(np.uint8)
    colors[codes == 0] = [70, 180, 95]
    colors[codes == 1] = [235, 165, 50]
    colors[codes == 2] = [230, 65, 65]
    colors[codes == 3] = [220, 70, 215]
    colors[codes == 4] = [55, 185, 220]
    return colors


def main() -> int:
    a = _args()
    d = a.i2_7_1_dir.expanduser().resolve()
    summary = json.loads((d / "i2_7_1_summary.json").read_text(encoding="utf-8"))
    if str(summary.get("status")) != "diagnosis_ready":
        raise RuntimeError("I2.7.1 preview requires DIAGNOSIS_READY evidence.")
    with np.load(Path(str(summary["artifacts"]["component_lineage_diagnostics_npz"])), allow_pickle=False) as p:
        points = np.asarray(p["baseline_points_reconstruction"], dtype=np.float64)
        rgb = np.asarray(p["baseline_colors_rgb"], dtype=np.uint8)
        codes = np.asarray(p["lineage_category_code"], dtype=np.int8)
    colors = rgb if a.color_mode == "rgb" else _lineage_colors(codes, rgb)
    points, colors = enforce_point_budget(points, colors, int(a.point_budget), seed=0)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    w = QMainWindow(); w.setWindowTitle("EIA TECH — I2.7.1 Detached Component Qualification")
    central = QWidget(); layout = QVBoxLayout(central)
    comp = summary["component_comparison"]
    layout.addWidget(QLabel(
        f"I2.7.1 | {a.color_mode} | {a.view} | baseline={comp['baseline']['largest_component_fraction']:.4f} | "
        f"promoted={comp['promoted']['largest_component_fraction']:.4f} | newly detached={comp['newly_detached_voxels']:,}"
    ))
    canvas = PointCloudCanvas(w); canvas.set_point_size(float(a.point_size)); canvas.set_grid_visible(True); canvas.update_points(points, colors); canvas.fit_scene()
    camera = getattr(canvas, "_camera", None)
    if camera is not None:
        camera.azimuth, camera.elevation = map(float, _VIEW_ANGLES[a.view])
        changed = getattr(camera, "view_changed", None)
        if callable(changed): changed()
    layout.addWidget(canvas.native, 1); w.setCentralWidget(central); w.resize(1366, 768); w.show()
    for _ in range(14): app.processEvents(); time.sleep(.02)
    output = a.output.expanduser().resolve() if a.output else d / f"i2_7_1_{a.color_mode}_{a.view}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not w.grab().save(str(output), "PNG"):
        canvas.close(); raise RuntimeError(f"Unable to save I2.7.1 preview: {output}")
    print(f"I2.7.1 component preview: {output}")
    print(f"rendered_points={len(points)} color_mode={a.color_mode} view={a.view}")
    if a.color_mode == "lineage":
        print("legend: green=promoted dominant orange=preexisting detached red=newly detached magenta=newly detached Core cyan=d12 removed")
    print("I2.7.1 diagnostics only: no geometry/filter/persistence/metric/measurement/mesh authority changed.")
    if a.show:
        code = app.exec(); canvas.close(); w.close(); return int(code)
    canvas.close(); w.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
