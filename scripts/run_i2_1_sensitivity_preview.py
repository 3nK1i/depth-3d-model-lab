#!/usr/bin/env python3
"""Preview I2.1 support-removal sensitivity without changing geometry."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i2-1-dir", type=Path, default=Path("/tmp/eia_i2_1_targeted_peripheral_diagnosis"))
    parser.add_argument("--scenario", default="early_suspect", help="segment name or view:<workspace_order>")
    parser.add_argument("--view", choices=tuple(_VIEW_ANGLES), default="oblique")
    parser.add_argument("--point-budget", type=int, default=250_000)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.point_budget <= 0 or args.point_size <= 0.0:
        parser.error("point budget and point size must be positive")
    return args


def _set_view(canvas: PointCloudCanvas, view: str) -> None:
    camera = getattr(canvas, "_camera", None)
    if camera is None:
        return
    camera.azimuth, camera.elevation = map(float, _VIEW_ANGLES[view])
    changed = getattr(camera, "view_changed", None)
    if callable(changed):
        changed()


def main() -> int:
    args = _args()
    directory = args.i2_1_dir.expanduser().resolve()
    summary = json.loads((directory / "i2_1_summary.json").read_text(encoding="utf-8"))
    path = Path(str(summary["artifacts"]["support_sensitivity_npz"])).expanduser().resolve()
    with np.load(path, allow_pickle=False) as payload:
        points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
        rgb = np.asarray(payload["colors_rgb"], dtype=np.uint8)
        peripheral = np.asarray(payload["is_peripheral"], dtype=bool)
        far = np.asarray(payload["is_far_peripheral"], dtype=bool)
        view_orders = np.asarray(payload["selected_workspace_orders"], dtype=np.int32)
        view_masks = np.asarray(payload["single_view_collapse_masks"], dtype=bool)
        segment_names = [str(v) for v in np.asarray(payload["segment_names"]).tolist()]
        segment_masks = np.asarray(payload["segment_collapse_masks"], dtype=bool)

    scenario = str(args.scenario)
    if scenario.startswith("view:"):
        order = int(scenario.split(":", 1)[1])
        rows = np.flatnonzero(view_orders == order)
        if len(rows) != 1:
            raise RuntimeError(f"I2.1 scenario references unavailable view order {order}")
        collapse = view_masks[int(rows[0])]
        label = f"view {order} removal support-collapse"
    else:
        if scenario not in segment_names:
            raise RuntimeError(f"Unknown I2.1 segment {scenario!r}; available: {', '.join(segment_names)}")
        collapse = segment_masks[segment_names.index(scenario)]
        label = f"segment {scenario} support-collapse"

    colors = np.asarray(rgb, dtype=np.uint8).copy()
    # De-emphasize retained geometry while preserving spatial context.
    colors = np.clip((colors.astype(np.float32) * 0.35) + 45.0, 0, 255).astype(np.uint8)
    colors[peripheral & ~collapse] = [230, 155, 55]
    colors[collapse] = [235, 70, 70]
    colors[collapse & far] = [230, 70, 220]

    points, colors = enforce_point_budget(points, colors, int(args.point_budget), seed=0)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow()
    window.setWindowTitle("Depth & 3D Model Lab — I2.1 Peripheral Support Sensitivity")
    central = QWidget()
    layout = QVBoxLayout(central)
    layout.addWidget(QLabel(
        f"I2.1 — {label} | red=loses >=2-view retention | magenta=far + loses retention | "
        "orange=peripheral but survives | diagnostic only"
    ))
    canvas = PointCloudCanvas(window)
    canvas.set_point_size(float(args.point_size))
    canvas.set_grid_visible(True)
    canvas.update_points(points, colors)
    canvas.fit_scene()
    _set_view(canvas, args.view)
    layout.addWidget(canvas.native, 1)
    window.setCentralWidget(central)
    window.resize(1366, 768)
    window.show()
    for _ in range(14):
        app.processEvents()
        time.sleep(0.02)

    safe = scenario.replace(":", "_").replace("/", "_")
    output = args.output.expanduser().resolve() if args.output else directory / f"i2_1_{safe}_{args.view}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close()
        raise RuntimeError(f"Unable to save I2.1 preview: {output}")
    print(f"I2.1 sensitivity preview: {output}")
    print(f"scenario={scenario} view={args.view} geometry_changed=False")
    if args.show:
        code = app.exec()
        canvas.close(); window.close()
        return int(code)
    canvas.close(); window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
