#!/usr/bin/env python3
"""Fixed-view I4.1.8 bound Dense RGB + retained-Core target orbit preview."""
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


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i4-1-8-dir", type=Path, default=Path("/tmp/eia_i4_1_8_production_binding"))
    p.add_argument("--geometry", choices=("extended", "core"), default="extended")
    p.add_argument("--view", choices=tuple(_VIEW_ANGLES), default="oblique")
    p.add_argument("--fit", choices=("all", "object"), default="object")
    p.add_argument("--point-budget", type=int, default=250_000)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root must be object: {path}")
    return payload


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
    directory = a.i4_1_8_dir.expanduser().resolve()
    summary = _read_json(directory / "i4_1_8_summary.json")
    if str(summary.get("phase")) != "I4.1.8":
        raise RuntimeError("I4.1.8 preview requires an I4.1.8 summary.")
    if str(summary.get("status")) not in ("ready_for_manual_orbit_review", "degraded"):
        raise RuntimeError("I4.1.8 preview requires a completed production-binding run.")

    f_summary = _read_json(Path(str(summary["artifacts"]["f_summary"])).expanduser().resolve())
    cloud_key = "extended_npz" if a.geometry == "extended" else "core_npz"
    cloud_path = Path(str(f_summary["artifacts"][cloud_key])).expanduser().resolve()
    diag_path = Path(str(summary["artifacts"]["final_orbit_diagnostics_npz"])).expanduser().resolve()
    with np.load(cloud_path, allow_pickle=False) as payload:
        points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
        colors = np.asarray(payload["colors_rgb"], dtype=np.uint8)
    with np.load(diag_path, allow_pickle=False) as payload:
        matrices = np.asarray(payload["production_selected_camera_matrices"], dtype=np.float64)
        source_indices = np.asarray(payload["production_selected_source_sequence_indices"], dtype=np.int64)
        target = np.asarray(payload["retained_core_target_reconstruction"], dtype=np.float64)
    camera_ids = tuple(f"source-{int(v)}" for v in source_indices.tolist())
    points, colors = enforce_point_budget(points, colors, int(a.point_budget), seed=0)

    metrics = summary["final_assessment"]["selected_orbit"]
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow(); window.setWindowTitle("EIA TECH — I4.1.8 Final Object-Orbit Review")
    central = QWidget(); layout = QVBoxLayout(central)
    layout.addWidget(QLabel(
        f"I4.1.8 | {a.geometry} | {len(points):,} rendered | {a.view} | fit={a.fit} | "
        f"bound cameras={len(camera_ids)} | coverage={metrics['azimuth_coverage_degrees']:.1f}° | "
        f"opposed={metrics['maximum_opposed_view_angle_degrees']:.1f}° | facing={metrics['target_facing_fraction']:.3f} | "
        "cyan=bound production orbit | magenta=retained-Core target | manual review"
    ))
    canvas = ReconstructionCanvas(window)
    canvas.set_grid_visible(True)
    canvas.set_dense_rgb_points(
        points,
        colors,
        scene_id=str(summary.get("artifacts", {}).get("persistent_scene", "i4-1-8")),
        representation=a.geometry,
        source_path=str(cloud_path),
    )
    canvas.update_sparse_landmarks(target.reshape(1, 3), colors_rgb=np.asarray([[255, 0, 255]], dtype=np.uint8), total_count=1)
    canvas.set_sparse_landmarks_visible(True)
    canvas.set_canonical_camera_trajectory(
        matrices_reconstruction=matrices,
        camera_ids=camera_ids,
        scene_id=str(summary.get("artifacts", {}).get("persistent_scene", "i4-1-8")),
        artifact_id="i4.1.8-bound-production-orbit",
        provider_id="offline-sfm-bound-production",
        total_count=len(camera_ids),
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
    output = a.output.expanduser().resolve() if a.output else directory / f"i4_1_8_{a.geometry}_{a.view}_{a.fit}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close(); raise RuntimeError(f"Unable to save I4.1.8 preview: {output}")
    print(f"I4.1.8 orbit preview: {output}")
    print(
        f"geometry={a.geometry} rendered_points={len(points)} bound_cameras={len(camera_ids)} "
        f"coverage={metrics['azimuth_coverage_degrees']:.1f}deg opposed={metrics['maximum_opposed_view_angle_degrees']:.1f}deg "
        f"facing={metrics['target_facing_fraction']:.3f} view={a.view} fit={a.fit}"
    )
    print("I4.1.8 manual review only: arbitrary SfM gauge; metric/measurement/mesh authority remain unpromoted.")
    if a.show:
        code = app.exec(); canvas.close(); window.close(); return int(code)
    canvas.close(); window.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
