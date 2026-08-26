#!/usr/bin/env python3
"""Fixed-view I4.1.1 Dense RGB + diagnostic camera-set preview."""
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

from app.widgets.viewport.vispy_canvas import ReconstructionCanvas  # noqa: E402
from core.geometry.sampling import enforce_point_budget  # noqa: E402

_VIEW_ANGLES = {"oblique": (32.0, 24.0), "front": (0.0, 0.0), "side": (90.0, 0.0), "top": (0.0, 89.0)}
_CAMERA_SETS = ("registered", "aligned", "rejected", "selected")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnosis-dir", type=Path, default=Path("/tmp/eia_i4_1_1_orbit_alignment_coverage_diagnosis"))
    p.add_argument("--geometry", choices=("extended", "core"), default="extended")
    p.add_argument("--camera-set", choices=_CAMERA_SETS, default="selected")
    p.add_argument("--view", choices=tuple(_VIEW_ANGLES), default="oblique")
    p.add_argument("--fit", choices=("all", "object"), default="all")
    p.add_argument("--point-budget", type=int, default=250_000)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def _object_fit(canvas: ReconstructionCanvas, points: np.ndarray) -> None:
    camera = getattr(canvas, "_camera", None)
    if camera is None or len(points) == 0:
        return
    lower = np.quantile(points, 0.02, axis=0)
    upper = np.quantile(points, 0.98, axis=0)
    center_recon = 0.5 * (lower + upper)
    center_viewer = np.asarray([center_recon[0], center_recon[2], -center_recon[1]], dtype=np.float64)
    diagonal = max(float(np.linalg.norm(upper - lower)), 0.25)
    camera.center = tuple(float(v) for v in center_viewer)
    camera.distance = max(diagonal * 1.45, 0.75)
    changed = getattr(camera, "view_changed", None)
    if callable(changed):
        changed()


def main() -> int:
    a = _args()
    directory = a.diagnosis_dir.expanduser().resolve()
    summary = json.loads((directory / "i4_1_1_summary.json").read_text(encoding="utf-8"))
    if str(summary.get("status")) != "diagnosis_ready":
        raise RuntimeError("I4.1.1 preview requires a completed diagnosis.")
    ext_path = Path(str(summary["artifacts"]["extended_npz"])).expanduser().resolve()
    diag_path = Path(str(summary["artifacts"]["diagnostics_npz"])).expanduser().resolve()
    with np.load(ext_path, allow_pickle=False) as p:
        points = np.asarray(p["points_reconstruction"], dtype=np.float64)
        colors = np.asarray(p["colors_rgb"], dtype=np.uint8)
        classes = np.asarray(p["class_code"], dtype=np.uint8).reshape(-1)
    with np.load(diag_path, allow_pickle=False) as p:
        matrices = np.asarray(p[f"{a.camera_set}_camera_matrices"], dtype=np.float64)
        ids = tuple(str(v) for v in np.asarray(p[f"{a.camera_set}_camera_ids"], dtype="U160").tolist())
        target = np.asarray(p["target_reconstruction"], dtype=np.float64)

    if a.geometry == "core":
        mask = classes == 3
        points = points[mask]
        colors = colors[mask]
    points, colors = enforce_point_budget(points, colors, int(a.point_budget), seed=0)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow(); window.setWindowTitle("EIA TECH — I4.1.1 Orbit Alignment / Coverage Diagnosis")
    central = QWidget(); layout = QVBoxLayout(central)
    metric = summary["orbit_diagnosis"].get(a.camera_set)
    if isinstance(metric, dict):
        metric_text = (
            f"coverage={metric['azimuth_coverage_degrees']:.1f}° | bins={metric['occupied_azimuth_bins']}/{metric['azimuth_bin_count']} | "
            f"opposed={metric['maximum_opposed_view_angle_degrees']:.1f}° | facing={metric['target_facing_fraction']:.3f}"
        )
    else:
        metric_text = "rejected cameras only (diagnostic set; no orbit gate)"
    layout.addWidget(QLabel(
        f"I4.1.1 | {a.geometry} | {len(points):,} rendered | cameras={a.camera_set}:{len(ids)} | {metric_text} | "
        "cyan=camera set | magenta=robust Core target | diagnosis only"
    ))
    canvas = ReconstructionCanvas(window)
    canvas.set_grid_visible(True)
    canvas.set_dense_rgb_points(
        points,
        colors,
        scene_id="i4.1.1-diagnosis",
        representation=a.geometry,
        source_path=str(ext_path),
    )
    canvas.update_sparse_landmarks(target.reshape(1, 3), colors_rgb=np.asarray([[255, 0, 255]], dtype=np.uint8), total_count=1)
    canvas.set_sparse_landmarks_visible(True)
    if len(ids):
        canvas.set_canonical_camera_trajectory(
            matrices_reconstruction=matrices,
            camera_ids=ids,
            scene_id="i4.1.1-diagnosis",
            artifact_id=f"i4.1.1-{a.camera_set}-cameras",
            provider_id="offline-sfm-diagnostic",
            total_count=len(ids),
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
        if callable(changed):
            changed()
    layout.addWidget(canvas.native, 1); window.setCentralWidget(central); window.resize(1366, 768); window.show()
    for _ in range(14): app.processEvents(); time.sleep(0.02)
    output = a.output.expanduser().resolve() if a.output else directory / f"i4_1_1_{a.geometry}_{a.camera_set}_{a.view}_{a.fit}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close(); raise RuntimeError(f"Unable to save I4.1.1 preview: {output}")
    print(f"I4.1.1 orbit diagnosis preview: {output}")
    print(f"geometry={a.geometry} rendered_points={len(points)} camera_set={a.camera_set} cameras={len(ids)} view={a.view} fit={a.fit}")
    print("I4.1.1 diagnosis only: no threshold/policy/geometry/authority promotion.")
    if a.show:
        code = app.exec(); canvas.close(); window.close(); return int(code)
    canvas.close(); window.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
