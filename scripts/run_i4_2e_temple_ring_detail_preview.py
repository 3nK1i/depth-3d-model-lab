#!/usr/bin/env python3
"""Render matched I4.2E pre-d12 vs post-d12 TempleRing detail previews."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.pop("DEPTH3D_TEST_FAKE_VISPY", None)

from app.widgets.viewport.vispy_canvas import PointCloudCanvas  # noqa: E402
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.mvs_visual_ab import MVS_VISUAL_AB_VIEWS  # noqa: E402

_DEFAULT_E = Path("/tmp/depth3d_i4_2_temple_ring/i4_2e_detail_qualification_attempt01")
_DEFAULT_OUT = Path("/tmp/depth3d_i4_2_temple_ring/i4_2e_detail_preview_attempt01")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i4-2e-dir", type=Path, default=_DEFAULT_E)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT)
    p.add_argument("--point-budget", type=int, default=500_000)
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--views", default="oblique,front,side,top")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()
    views = [item.strip().lower() for item in str(args.views).split(",") if item.strip()]
    if not views or len(set(views)) != len(views) or any(view not in MVS_VISUAL_AB_VIEWS for view in views):
        p.error("--views must contain unique names from: " + ",".join(MVS_VISUAL_AB_VIEWS))
    if int(args.point_budget) <= 0 or float(args.point_size) <= 0.0:
        p.error("point budget and point size must be positive")
    args.views = views
    return args


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReconstructionValidationError(f"I4.2E preview JSON must be an object: {path}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _shared_frame(points: np.ndarray) -> tuple[tuple[float, float, float], float, dict]:
    xyz = np.asarray(points, dtype=np.float64)
    lo = np.quantile(xyz, 0.02, axis=0)
    hi = np.quantile(xyz, 0.98, axis=0)
    center = 0.5 * (lo + hi)
    diagonal = float(np.linalg.norm(hi - lo))
    if not np.isfinite(diagonal) or diagonal <= 0.0:
        raise ReconstructionValidationError("I4.2E preview robust framing is degenerate.")
    center_viewer = (float(center[0]), float(center[2]), float(-center[1]))
    distance = max(diagonal * 1.45, 0.75)
    return center_viewer, distance, {
        "basis": "pre_d12_extended_q02_q98_shared_framing_only",
        "lower_reconstruction": lo.tolist(),
        "upper_reconstruction": hi.tolist(),
        "center_reconstruction": center.tolist(),
        "center_viewer": list(center_viewer),
        "robust_diagonal": diagonal,
        "camera_distance": distance,
        "independent_auto_fit": False,
        "metric_scale_claimed": False,
    }


def _camera(canvas: PointCloudCanvas, center: tuple[float, float, float], distance: float, view: str) -> None:
    camera = getattr(canvas, "_camera", None)
    if camera is None:
        raise RuntimeError("I4.2E preview requires PointCloudCanvas camera access.")
    camera.center = center
    camera.distance = float(distance)
    camera.fov = 45.0
    camera.azimuth, camera.elevation = map(float, MVS_VISUAL_AB_VIEWS[view])
    changed = getattr(camera, "view_changed", None)
    if callable(changed):
        changed()


def _render(
    *,
    app: QApplication,
    points: np.ndarray,
    colors: np.ndarray,
    label_text: str,
    view: str,
    output: Path,
    center: tuple[float, float, float],
    distance: float,
    point_size: float,
) -> None:
    window = QMainWindow()
    central = QWidget()
    layout = QVBoxLayout(central)
    label = QLabel(label_text)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(label)
    canvas = PointCloudCanvas(window)
    canvas.set_point_size(float(point_size))
    canvas.set_grid_visible(True)
    canvas.update_points(points, colors)
    _camera(canvas, center, distance, view)
    layout.addWidget(canvas.native, 1)
    window.setCentralWidget(central)
    window.resize(1366, 768)
    window.show()
    for _ in range(18):
        app.processEvents()
        time.sleep(0.02)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close(); window.close()
        raise RuntimeError(f"Could not save I4.2E preview: {output}")
    canvas.close(); window.close(); app.processEvents()


def _combine(left: Path, right: Path, output: Path) -> None:
    a, b = QImage(str(left)), QImage(str(right))
    if a.isNull() or b.isNull() or a.height() != b.height():
        raise RuntimeError("I4.2E matched preview images are invalid.")
    image = QImage(a.width() + b.width(), a.height(), QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.black)
    painter = QPainter(image)
    painter.drawImage(0, 0, a)
    painter.drawImage(a.width(), 0, b)
    painter.end()
    if not image.save(str(output), "PNG"):
        raise RuntimeError(f"Could not save I4.2E A/B sheet: {output}")


def main() -> int:
    args = _args()
    source = args.i4_2e_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    try:
        if output.exists() and any(output.iterdir()):
            raise ReconstructionValidationError("I4.2E preview output must be empty or absent.")
        summary_path = source / "i4_2e_summary.json"
        summary = _read_json(summary_path)
        if summary.get("phase") != "I4.2E" or summary.get("status") != "ready_for_manual_review" or summary.get("pass") is not True:
            raise ReconstructionValidationError("I4.2E preview requires READY_FOR_MANUAL_REVIEW qualification evidence.")
        authority = summary.get("authority", {})
        if authority.get("d12_counterfactual_only") is not True or authority.get("production_filter_policy_changed") is not False:
            raise ReconstructionValidationError("I4.2E preview source does not preserve the counterfactual-only policy lock.")
        analysis = Path(str(summary["artifacts"]["analysis_npz"])).resolve()
        if _sha(analysis) != str(summary["artifact_sha256"]["analysis_npz"]):
            raise ReconstructionValidationError("I4.2E analysis NPZ changed after qualification.")
        with np.load(analysis, allow_pickle=False) as payload:
            points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
            colors = np.asarray(payload["colors_rgb"], dtype=np.uint8)
            reject = np.asarray(payload["d12_reject_mask"], dtype=bool)
        if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape or reject.shape != (len(points),):
            raise ReconstructionValidationError("I4.2E preview analysis arrays are malformed.")
        if not np.isfinite(points).all():
            raise ReconstructionValidationError("I4.2E preview geometry is non-finite.")

        # Derive the deterministic display subset exactly once so PRE, POST,
        # and rejection overlays share identical source indices.
        if len(points) <= int(args.point_budget):
            sample_idx = np.arange(len(points), dtype=np.int64)
        else:
            rng = np.random.default_rng(0)
            sample_idx = np.sort(
                rng.choice(len(points), size=int(args.point_budget), replace=False)
            )
        pre_points = points[sample_idx]
        pre_colors = colors[sample_idx]
        sample_reject = reject[sample_idx]
        post_keep = ~sample_reject
        post_points = pre_points[post_keep]
        post_colors = pre_colors[post_keep]
        loss_colors = pre_colors.copy()
        # Diagnostic-only highlight: rejected d12 counterfactual points in red.
        loss_colors[sample_reject] = np.asarray([255, 48, 48], dtype=np.uint8)

        center, distance, framing = _shared_frame(points)
        output.mkdir(parents=True, exist_ok=True)
        renders = output / "renders"
        renders.mkdir()
        app = QApplication.instance() or QApplication(sys.argv[:1])
        artifacts: dict[str, dict] = {}
        for view in args.views:
            pre = renders / f"i4_2e_{view}_pre_d12.png"
            post = renders / f"i4_2e_{view}_post_d12.png"
            ab = renders / f"i4_2e_{view}_ab.png"
            loss = renders / f"i4_2e_{view}_loss_overlay.png"
            common = f"view={view} | SHARED CAMERA | arbitrary SfM gauge | manual detail review"
            _render(
                app=app, points=pre_points, colors=pre_colors,
                label_text=f"LEFT — PRE-d12 MVS-derived Extended | {common}", view=view, output=pre,
                center=center, distance=distance, point_size=float(args.point_size),
            )
            _render(
                app=app, points=post_points, colors=post_colors,
                label_text=f"RIGHT — POST-d12 exact predicate counterfactual | {common}", view=view, output=post,
                center=center, distance=distance, point_size=float(args.point_size),
            )
            _combine(pre, post, ab)
            _render(
                app=app, points=pre_points, colors=loss_colors,
                label_text=f"D12 REJECTION OVERLAY — red=rejected | {common}", view=view, output=loss,
                center=center, distance=distance, point_size=float(args.point_size),
            )
            artifacts[view] = {
                "pre_d12": str(pre), "post_d12": str(post), "ab": str(ab), "loss_overlay": str(loss),
                "sha256": {"pre_d12": _sha(pre), "post_d12": _sha(post), "ab": _sha(ab), "loss_overlay": _sha(loss)},
            }
            print(f"I4.2E {view}: {ab}")

        preview_summary = {
            "schema_version": 1,
            "phase": "I4.2E-preview",
            "status": "ready_for_manual_review",
            "source_i4_2e_summary": str(summary_path),
            "source_i4_2e_summary_sha256": _sha(summary_path),
            "shared_framing": framing,
            "point_budget": int(args.point_budget),
            "pre_d12_source_points": int(len(points)),
            "sampled_points": int(len(pre_points)),
            "sampled_d12_rejected": int(np.count_nonzero(sample_reject)),
            "same_sample_indices_pre_post": True,
            "same_camera_pre_post": True,
            "independent_auto_fit": False,
            "automatic_winner_selected": False,
            "manual_visual_review_required": True,
            "production_filter_policy_changed": False,
            "i4_3_provider_architecture_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
            "renders": artifacts,
        }
        path = _write_json(output / "i4_2e_preview_summary.json", preview_summary)
        print("I4.2E matched detail preview: READY FOR MANUAL REVIEW")
        print(f"sampled={len(pre_points)} d12_rejected_in_sample={int(np.count_nonzero(sample_reject))}")
        print(f"summary={path}")

        if args.show:
            image = QImage(str(Path(artifacts[args.views[0]]["ab"])))
            window = QMainWindow()
            window.setWindowTitle("Depth & 3D Model Lab — I4.2E TempleRing d12 Detail A/B")
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setPixmap(QPixmap.fromImage(image).scaled(1320, 700, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            window.setCentralWidget(label)
            window.resize(1366, 768)
            window.show()
            return int(app.exec())
        return 0
    except (OSError, KeyError, json.JSONDecodeError, ReconstructionValidationError) as exc:
        print(f"I4.2E matched detail preview: FAIL — {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
