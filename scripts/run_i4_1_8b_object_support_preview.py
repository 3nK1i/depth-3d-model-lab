#!/usr/bin/env python3
"""Visual review of I4.1.8B isolated retained-Core object support.

Full retained Core is rendered in RGB.  The independently isolated support is
shown as bright green sparse points.  Marker colors:
  white   = I4.1.8B isolated-support geometric-median candidate
  magenta = current whole-Core median from I4.1.8A
  yellow  = production trajectory-circle target from I4.1.8A
  cyan    = bound production cameras

This remains diagnosis/isolation only; no displayed target is promoted.
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
    p.add_argument("--i4-1-8b-dir", type=Path, default=Path("/tmp/eia_i4_1_8b_retained_core_object_support"))
    p.add_argument("--view", choices=tuple(_VIEW_ANGLES), default="oblique")
    p.add_argument("--fit", choices=("all", "core", "support"), default="all")
    p.add_argument("--point-budget", type=int, default=250_000)
    p.add_argument("--support-budget", type=int, default=25_000)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root must be object: {path}")
    return payload


def _fit_points(canvas: ReconstructionCanvas, points: np.ndarray) -> None:
    camera = getattr(canvas, "_camera", None)
    if camera is None or len(points) == 0:
        return
    lower = np.quantile(points, 0.02, axis=0); upper = np.quantile(points, 0.98, axis=0)
    center = 0.5 * (lower + upper)
    viewer_center = np.asarray([center[0], center[2], -center[1]], dtype=np.float64)
    diagonal = max(float(np.linalg.norm(upper - lower)), 0.25)
    camera.center = tuple(float(v) for v in viewer_center)
    camera.distance = max(diagonal * 1.45, 0.75)
    changed = getattr(camera, "view_changed", None)
    if callable(changed):
        changed()


def main() -> int:
    a = _args(); directory = a.i4_1_8b_dir.expanduser().resolve()
    summary = _read_json(directory / "i4_1_8b_summary.json")
    if str(summary.get("phase")) != "I4.1.8B":
        raise RuntimeError("I4.1.8B preview requires an I4.1.8B summary.")
    artifact = Path(str(summary["artifacts"]["object_support_diagnostics_npz"])).expanduser().resolve()
    core_path = Path(str(summary["source_artifacts"]["retained_core_npz"])).expanduser().resolve()
    a_summary = _read_json(Path(str(summary["source_artifacts"]["i4_1_8a_summary"])).expanduser().resolve())
    with np.load(core_path, allow_pickle=False) as p:
        core_points = np.asarray(p["points_reconstruction"], dtype=np.float64)
        core_colors = np.asarray(p["colors_rgb"], dtype=np.uint8)
    with np.load(artifact, allow_pickle=False) as p:
        support_points = np.asarray(p["isolated_core_points_reconstruction"], dtype=np.float64)
        matrices = np.asarray(p["production_selected_camera_matrices"], dtype=np.float64)
        sources = np.asarray(p["production_selected_source_sequence_indices"], dtype=np.int64)
    core_points, core_colors = enforce_point_budget(core_points, core_colors, int(a.point_budget), seed=0)
    if len(support_points) > int(a.support_budget):
        idx = np.rint(np.linspace(0, len(support_points)-1, int(a.support_budget))).astype(np.int64)
        support_render = support_points[idx]
    else:
        support_render = support_points

    sparse_points: list[np.ndarray] = []
    sparse_colors: list[np.ndarray] = []
    if len(support_render):
        sparse_points.extend(np.asarray(support_render, dtype=np.float64))
        sparse_colors.extend(np.tile(np.asarray([[0, 255, 80]], dtype=np.uint8), (len(support_render), 1)))
    selected = summary.get("selected_support")
    if isinstance(selected, dict):
        sparse_points.append(np.asarray(selected["target_reconstruction"], dtype=np.float64))
        sparse_colors.append(np.asarray([255, 255, 255], dtype=np.uint8))
    targets_a = a_summary.get("targets", {})
    for name, color in (
        ("current_whole_core_coordinate_median", [255, 0, 255]),
        ("production_bound_trajectory_circle", [255, 220, 0]),
    ):
        record = targets_a.get(name)
        if isinstance(record, dict):
            sparse_points.append(np.asarray(record["target_reconstruction"], dtype=np.float64))
            sparse_colors.append(np.asarray(color, dtype=np.uint8))

    sparse_xyz = np.asarray(sparse_points, dtype=np.float64).reshape(-1, 3) if sparse_points else np.empty((0,3), dtype=np.float64)
    sparse_rgb = np.asarray(sparse_colors, dtype=np.uint8).reshape(-1, 3) if sparse_colors else np.empty((0,3), dtype=np.uint8)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow(); window.setWindowTitle("Depth & 3D Model Lab — I4.1.8B Retained-Core Object Support")
    central = QWidget(); layout = QVBoxLayout(central)
    diag = summary["diagnosis"]
    posthoc = selected.get("posthoc_orbit_diagnosis", {}).get("current_metric", {}) if isinstance(selected, dict) else {}
    layout.addWidget(QLabel(
        f"I4.1.8B | Core={len(core_points):,} rendered | isolated={len(support_points):,} | cameras={len(matrices)} | "
        f"{a.view} | fit={a.fit} | class={diag['classification']} | "
        f"posthoc={float(posthoc.get('azimuth_coverage_degrees',0.0)):.1f}° | "
        "green=isolated Core support | white=isolated-support target | magenta=whole-Core median | yellow=trajectory circle | cyan=cameras | isolation only"
    ))
    canvas = ReconstructionCanvas(window); canvas.set_grid_visible(True)
    canvas.set_dense_rgb_points(
        core_points, core_colors, scene_id="i4-1-8b-object-support", representation="core", source_path=str(core_path)
    )
    if len(sparse_xyz):
        canvas.update_sparse_landmarks(sparse_xyz, colors_rgb=sparse_rgb, total_count=len(sparse_xyz))
        canvas.set_sparse_landmarks_visible(True)
    camera_ids = tuple(f"source-{int(v)}" for v in sources.tolist())
    canvas.set_canonical_camera_trajectory(
        matrices_reconstruction=matrices,
        camera_ids=camera_ids,
        scene_id="i4-1-8b-object-support",
        artifact_id="i4.1.8b-bound-production-orbit",
        provider_id="offline-sfm-bound-production",
        total_count=len(camera_ids),
    )
    canvas.set_canonical_camera_frustums_visible(True); canvas.set_canonical_camera_trajectory_visible(True)
    if a.fit == "all":
        canvas.fit_scene()
    elif a.fit == "support" and len(support_points):
        _fit_points(canvas, support_points)
    else:
        _fit_points(canvas, core_points)
    camera = getattr(canvas, "_camera", None)
    if camera is not None:
        camera.azimuth, camera.elevation = map(float, _VIEW_ANGLES[a.view])
        changed = getattr(camera, "view_changed", None)
        if callable(changed): changed()
    layout.addWidget(canvas.native, 1); window.setCentralWidget(central); window.resize(1366,768); window.show()
    for _ in range(14): app.processEvents(); time.sleep(0.02)
    output = a.output.expanduser().resolve() if a.output else directory / f"i4_1_8b_{a.view}_{a.fit}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close(); raise RuntimeError(f"Unable to save I4.1.8B preview: {output}")
    print(f"I4.1.8B object-support preview: {output}")
    print("green=isolated retained Core support | white=isolation target | magenta=whole-Core median | yellow=trajectory circle")
    print("I4.1.8B preview is isolation-only; no target/metric/measurement/mesh authority is promoted.")
    if a.show:
        code = app.exec(); canvas.close(); window.close(); return int(code)
    canvas.close(); window.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
