#!/usr/bin/env python3
"""Fixed-view preview for I2.7 promoted-scene final quality review."""
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
    p.add_argument("--i2-7-dir", type=Path, default=Path("/tmp/eia_i2_7_final_requalification"))
    p.add_argument("--representation", choices=("extended", "core"), default="extended")
    p.add_argument("--color-mode", choices=("rgb", "region", "support", "confidence", "dispersion"), default="rgb")
    p.add_argument("--view", choices=tuple(_VIEW_ANGLES), default="oblique")
    p.add_argument("--fit", choices=("full", "robust"), default="full")
    p.add_argument("--point-budget", type=int, default=250_000)
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def _scalar(values: np.ndarray, invert: bool = False) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(data)
    norm = np.zeros(len(data), dtype=np.float64)
    if np.any(finite):
        lo, hi = np.quantile(data[finite], [0.05, 0.95])
        if hi <= lo: hi = lo + 1.0
        norm[finite] = np.clip((data[finite] - lo) / (hi - lo), 0.0, 1.0)
    if invert: norm = 1.0 - norm
    return np.clip(np.rint(np.column_stack([
        np.clip(2.0 * norm - 0.2, 0, 1),
        np.clip(1.7 * norm, 0, 1),
        np.clip(1.2 - 1.4 * norm, 0, 1),
    ]) * 255), 0, 255).astype(np.uint8)


def _region(peripheral: np.ndarray, far: np.ndarray, weak: np.ndarray) -> np.ndarray:
    c = np.tile(np.asarray([[80, 205, 110]], dtype=np.uint8), (len(peripheral), 1))
    c[peripheral] = [235, 190, 55]
    c[weak] = [235, 120, 55]
    c[far] = [225, 70, 70]
    return c


def _fit_robust(canvas: PointCloudCanvas, summary: dict) -> None:
    env = summary["spatial"]["core_anchor_envelope"]
    lower, upper = np.asarray(env["lower"], float), np.asarray(env["upper"], float)
    camera = getattr(canvas, "_camera", None)
    if camera is None: return
    camera.center = tuple(((lower + upper) * 0.5).tolist())
    camera.distance = max(float(np.linalg.norm(upper - lower)) * 1.35, 1.5)


def main() -> int:
    a = _args(); d = a.i2_7_dir.expanduser().resolve()
    s = json.loads((d / "i2_7_summary.json").read_text(encoding="utf-8"))
    if not bool(s.get("automated_pass")): raise RuntimeError("I2.7 preview requires automated_pass=True.")
    with np.load(Path(str(s["artifacts"]["spatial_diagnostics_npz"])), allow_pickle=False) as p:
        points = np.asarray(p["points_reconstruction"], float); rgb = np.asarray(p["colors_rgb"], np.uint8)
        support = np.asarray(p["view_observation_counts"], float); confidence = np.asarray(p["filter_confidence"], float)
        dispersion = np.asarray(p["p90_dispersion_voxels"], float); is_core = np.asarray(p["is_core"], bool)
        peripheral = np.asarray(p["is_peripheral"], bool); far = np.asarray(p["is_far_peripheral"], bool); weak = np.asarray(p["is_weak_peripheral"], bool)
    mask = np.ones(len(points), bool) if a.representation == "extended" else is_core
    points, rgb, support, confidence, dispersion = points[mask], rgb[mask], support[mask], confidence[mask], dispersion[mask]
    peripheral, far, weak = peripheral[mask], far[mask], weak[mask]
    colors = rgb if a.color_mode == "rgb" else (_region(peripheral, far, weak) if a.color_mode == "region" else _scalar({"support": support, "confidence": confidence, "dispersion": dispersion}[a.color_mode], invert=a.color_mode == "dispersion"))
    points, colors = enforce_point_budget(points, colors, int(a.point_budget), seed=0)
    app = QApplication.instance() or QApplication(sys.argv[:1]); w = QMainWindow(); w.setWindowTitle("EIA TECH — I2.7 Final Full-Video Quality Review")
    central = QWidget(); layout = QVBoxLayout(central)
    layout.addWidget(QLabel(f"I2.7 | {a.representation} | {len(points):,} rendered | {a.color_mode} | {a.view} | {a.fit} | full/robust={float(s['spatial']['promoted_full_to_robust_ratio']):.3f} | final manual review"))
    canvas = PointCloudCanvas(w); canvas.set_point_size(float(a.point_size)); canvas.set_grid_visible(True); canvas.update_points(points, colors)
    canvas.fit_scene() if a.fit == "full" else _fit_robust(canvas, s)
    camera = getattr(canvas, "_camera", None)
    if camera is not None:
        camera.azimuth, camera.elevation = map(float, _VIEW_ANGLES[a.view]); changed = getattr(camera, "view_changed", None)
        if callable(changed): changed()
    layout.addWidget(canvas.native, 1); w.setCentralWidget(central); w.resize(1366, 768); w.show()
    for _ in range(14): app.processEvents(); time.sleep(.02)
    output = a.output.expanduser().resolve() if a.output else d / f"i2_7_{a.representation}_{a.color_mode}_{a.view}_{a.fit}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not w.grab().save(str(output), "PNG"): canvas.close(); raise RuntimeError(f"Unable to save I2.7 preview: {output}")
    print(f"I2.7 fixed-view preview: {output}")
    print(f"representation={a.representation} rendered_points={len(points)} color_mode={a.color_mode} view={a.view} fit={a.fit}")
    print("I2.7 review only: promoted F/G stays authoritative; metric/measurement/mesh authority remains unpromoted.")
    if a.show:
        code = app.exec(); canvas.close(); w.close(); return int(code)
    canvas.close(); w.close(); return 0


if __name__ == "__main__": raise SystemExit(main())
