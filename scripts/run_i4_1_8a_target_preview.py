#!/usr/bin/env python3
"""Visual I4.1.8A target comparison over retained Core + bound cameras.

Marker colors:
  magenta = current whole-Core coordinate-wise median
  yellow  = production bound trajectory-circle center
  green   = retained-Core camera-attention candidate (when available)
  orange  = production optical-axis focus

This is manual diagnosis only; none of the displayed targets are promoted by
this script.
"""
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
    p.add_argument("--i4-1-8a-dir", type=Path, default=Path("/tmp/eia_i4_1_8a_retained_core_target_diagnosis"))
    p.add_argument("--view", choices=tuple(_VIEW_ANGLES), default="oblique")
    p.add_argument("--fit", choices=("all", "core"), default="all")
    p.add_argument("--point-budget", type=int, default=250_000)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root must be object: {path}")
    return payload


def _core_fit(canvas: ReconstructionCanvas, points: np.ndarray) -> None:
    camera = getattr(canvas, "_camera", None)
    if camera is None or len(points) == 0:
        return
    lower = np.quantile(points, 0.02, axis=0)
    upper = np.quantile(points, 0.98, axis=0)
    center = 0.5 * (lower + upper)
    viewer_center = np.asarray([center[0], center[2], -center[1]], dtype=np.float64)
    diagonal = max(float(np.linalg.norm(upper - lower)), 0.25)
    camera.center = tuple(float(v) for v in viewer_center)
    camera.distance = max(diagonal * 1.45, 0.75)
    changed = getattr(camera, "view_changed", None)
    if callable(changed):
        changed()


def main() -> int:
    a = _args()
    directory = a.i4_1_8a_dir.expanduser().resolve()
    summary = _read_json(directory / "i4_1_8a_summary.json")
    if str(summary.get("phase")) != "I4.1.8A" or str(summary.get("status")) != "diagnosis_ready":
        raise RuntimeError("I4.1.8A preview requires a completed diagnosis_ready summary.")

    core_path = Path(str(summary["source_artifacts"]["retained_core_npz"])).expanduser().resolve()
    diag_path = Path(str(summary["artifacts"]["diagnostic_npz"])).expanduser().resolve()
    with np.load(core_path, allow_pickle=False) as payload:
        points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
        colors = np.asarray(payload["colors_rgb"], dtype=np.uint8)
    with np.load(diag_path, allow_pickle=False) as payload:
        matrices = np.asarray(payload["production_selected_camera_matrices"], dtype=np.float64)
        sources = np.asarray(payload["production_selected_source_sequence_indices"], dtype=np.int64)

    points, colors = enforce_point_budget(points, colors, int(a.point_budget), seed=0)
    target_order = [
        ("current_whole_core_coordinate_median", [255, 0, 255]),
        ("production_bound_trajectory_circle", [255, 220, 0]),
        ("retained_core_camera_attention_candidate", [0, 255, 80]),
        ("production_optical_axis_focus", [255, 135, 0]),
    ]
    target_points: list[np.ndarray] = []
    target_colors: list[list[int]] = []
    active_names: list[str] = []
    for name, color in target_order:
        record = summary.get("targets", {}).get(name)
        if isinstance(record, dict):
            target_points.append(np.asarray(record["target_reconstruction"], dtype=np.float64))
            target_colors.append(color)
            active_names.append(name)
    if not target_points:
        raise RuntimeError("I4.1.8A preview found no target markers.")

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow(); window.setWindowTitle("Depth & 3D Model Lab — I4.1.8A Retained-Core Target Diagnosis")
    central = QWidget(); layout = QVBoxLayout(central)
    diagnosis = summary["diagnosis"]
    plane = summary["retained_core"]["dominant_plane"]
    layout.addWidget(QLabel(
        f"I4.1.8A | {len(points):,} Core points | cameras={len(matrices)} | {a.view} | fit={a.fit} | "
        f"class={diagnosis['classification']} | dominant-plane={plane['core_inlier_fraction']:.3f} | "
        "magenta=current Core median | yellow=trajectory circle | green=Core+camera attention | orange=optical focus | diagnosis only"
    ))
    canvas = ReconstructionCanvas(window)
    canvas.set_grid_visible(True)
    canvas.set_dense_rgb_points(
        points,
        colors,
        scene_id="i4-1-8a-retained-core-diagnosis",
        representation="core",
        source_path=str(core_path),
    )
    canvas.update_sparse_landmarks(
        np.asarray(target_points, dtype=np.float64),
        colors_rgb=np.asarray(target_colors, dtype=np.uint8),
        total_count=len(target_points),
    )
    canvas.set_sparse_landmarks_visible(True)
    camera_ids = tuple(f"source-{int(v)}" for v in sources.tolist())
    canvas.set_canonical_camera_trajectory(
        matrices_reconstruction=matrices,
        camera_ids=camera_ids,
        scene_id="i4-1-8a-retained-core-diagnosis",
        artifact_id="i4.1.8a-bound-production-orbit",
        provider_id="offline-sfm-bound-production",
        total_count=len(camera_ids),
    )
    canvas.set_canonical_camera_frustums_visible(True)
    canvas.set_canonical_camera_trajectory_visible(True)
    if a.fit == "all":
        canvas.fit_scene()
    else:
        _core_fit(canvas, points)
    camera = getattr(canvas, "_camera", None)
    if camera is not None:
        camera.azimuth, camera.elevation = map(float, _VIEW_ANGLES[a.view])
        changed = getattr(camera, "view_changed", None)
        if callable(changed):
            changed()

    layout.addWidget(canvas.native, 1); window.setCentralWidget(central); window.resize(1366, 768); window.show()
    for _ in range(14):
        app.processEvents(); time.sleep(0.02)
    output = a.output.expanduser().resolve() if a.output else directory / f"i4_1_8a_{a.view}_{a.fit}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close(); raise RuntimeError(f"Unable to save I4.1.8A preview: {output}")
    print(f"I4.1.8A target preview: {output}")
    print("markers=" + ",".join(active_names))
    print("I4.1.8A preview is diagnosis-only; no target/metric/measurement/mesh authority is promoted.")
    if a.show:
        code = app.exec(); canvas.close(); window.close(); return int(code)
    canvas.close(); window.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
