#!/usr/bin/env python3
"""Fixed-view I4.1 Dense RGB + selected-camera orbit preview."""
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
    p.add_argument("--i4-dir", type=Path, default=Path("/tmp/eia_i4_object_orbit_campaign"))
    p.add_argument("--geometry", choices=("extended", "core"), default="extended")
    p.add_argument("--view", choices=tuple(_VIEW_ANGLES), default="oblique")
    p.add_argument("--fit", choices=("all", "object"), default="all")
    p.add_argument("--point-budget", type=int, default=250_000)
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def _object_fit(canvas: PointCloudCanvas, points: np.ndarray) -> None:
    camera = getattr(canvas, "_camera", None)
    if camera is None or len(points) == 0:
        return
    lower = np.quantile(points, 0.02, axis=0)
    upper = np.quantile(points, 0.98, axis=0)
    # PointCloudCanvas converts reconstruction (x,y,z) -> viewer (x,z,-y).
    center_recon = 0.5 * (lower + upper)
    center_viewer = np.asarray([center_recon[0], center_recon[2], -center_recon[1]], dtype=np.float64)
    extent = upper - lower
    diagonal = max(float(np.linalg.norm(extent)), 0.25)
    camera.center = tuple(float(v) for v in center_viewer)
    camera.distance = max(diagonal * 1.45, 0.75)
    changed = getattr(camera, "view_changed", None)
    if callable(changed):
        changed()


def main() -> int:
    a = _args()
    directory = a.i4_dir.expanduser().resolve()
    summary = json.loads((directory / "i4_1_summary.json").read_text(encoding="utf-8"))
    if str(summary.get("status")) not in ("ready_for_manual_orbit_review", "degraded"):
        raise RuntimeError("I4.1 preview requires a completed orbit campaign.")
    ext_path = Path(str(summary["artifacts"]["extended_npz"])).expanduser().resolve()
    diag_path = Path(str(summary["artifacts"]["orbit_diagnostics_npz"])).expanduser().resolve()
    with np.load(ext_path, allow_pickle=False) as p:
        points = np.asarray(p["points_reconstruction"], dtype=np.float64)
        colors = np.asarray(p["colors_rgb"], dtype=np.uint8)
        classes = np.asarray(p["class_code"], dtype=np.uint8).reshape(-1)
    with np.load(diag_path, allow_pickle=False) as p:
        selected_matrices = np.asarray(p["selected_camera_matrices"], dtype=np.float64)
        selected_ids = tuple(str(v) for v in np.asarray(p["selected_camera_ids"], dtype="U160").tolist())
        target = np.asarray(p["target_reconstruction"], dtype=np.float64)

    if a.geometry == "core":
        mask = classes == 3
        points = points[mask]
        colors = colors[mask]
    points, colors = enforce_point_budget(points, colors, int(a.point_budget), seed=0)

    metrics = summary["orbit_assessment"]["selected_orbit"]
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow(); window.setWindowTitle("Depth & 3D Model Lab — I4.1 Object / Connector Orbit Review")
    central = QWidget(); layout = QVBoxLayout(central)
    layout.addWidget(QLabel(
        f"I4.1 | {a.geometry} | {len(points):,} rendered | {a.view} | fit={a.fit} | "
        f"selected cameras={len(selected_ids)} | coverage={metrics['azimuth_coverage_degrees']:.1f}° | "
        f"opposed={metrics['maximum_opposed_view_angle_degrees']:.1f}° | facing={metrics['target_facing_fraction']:.3f} | "
        "cyan=selected camera orbit | magenta=robust Core target | review only"
    ))
    canvas = PointCloudCanvas(window)
    canvas.set_point_size(float(a.point_size))
    canvas.set_grid_visible(True)
    canvas.update_points(points, colors)
    canvas.update_sparse_landmarks(target.reshape(1, 3), colors_rgb=np.asarray([[255, 0, 255]], dtype=np.uint8), total_count=1)
    canvas.set_sparse_landmarks_visible(True)
    canvas.set_canonical_camera_trajectory(
        matrices_reconstruction=selected_matrices,
        camera_ids=selected_ids,
        scene_id=str(summary.get("scene", {}).get("scene_id", "i4-orbit")) or "i4-orbit",
        artifact_id="i4.1-selected-orbit",
        provider_id="offline-sfm-selected-qualified",
        total_count=len(selected_ids),
    )
    canvas.set_canonical_camera_frustums_visible(True)
    canvas.set_canonical_camera_trajectory_visible(True)
    if a.fit == "all":
        canvas.fit_scene()
    else:
        _object_fit(canvas, points)
    camera = getattr(canvas, "_camera", None)
    if camera is not None:
        camera.azimuth, camera.elevation = map(float, _VIEW_ANGLES[a.view])
        changed = getattr(camera, "view_changed", None)
        if callable(changed): changed()

    layout.addWidget(canvas.native, 1); window.setCentralWidget(central); window.resize(1366, 768); window.show()
    for _ in range(14): app.processEvents(); time.sleep(0.02)
    output = a.output.expanduser().resolve() if a.output else directory / f"i4_1_{a.geometry}_{a.view}_{a.fit}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close(); raise RuntimeError(f"Unable to save I4.1 preview: {output}")
    print(f"I4.1 orbit preview: {output}")
    print(
        f"geometry={a.geometry} rendered_points={len(points)} selected_cameras={len(selected_ids)} "
        f"coverage={metrics['azimuth_coverage_degrees']:.1f}deg opposed={metrics['maximum_opposed_view_angle_degrees']:.1f}deg "
        f"facing={metrics['target_facing_fraction']:.3f} view={a.view} fit={a.fit}"
    )
    print("I4.1 review only: arbitrary SfM gauge; metric/measurement/mesh authority remain unpromoted.")
    if a.show:
        code = app.exec(); canvas.close(); window.close(); return int(code)
    canvas.close(); window.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
