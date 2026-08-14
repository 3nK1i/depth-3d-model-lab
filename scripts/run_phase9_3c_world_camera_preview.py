#!/usr/bin/env python3
"""Real-VisPy preview for Phase 9.3C canonical world frame, cameras and trajectory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

# This preview must exercise the actual VisPy/OpenGL ReconstructionCanvas.
os.environ.pop("DEPTH3D_TEST_FAKE_VISPY", None)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.reconstruction_canonical_cameras import build_canonical_camera_trajectory_snapshot
from app.widgets.reconstruction_page import ReconstructionPage
from core.reconstruction import persistent_scene_from_pipeline_result
from core.reconstruction.orchestration import ReconstructionArtifactKind
from tests.reconstruction_f5_helpers import run_f5_fixture


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/phase9_3c_world_camera_preview.png"),
        help="PNG screenshot path",
    )
    parser.add_argument("--show", action="store_true", help="Keep the real VisPy preview open.")
    return parser


def _write_colored_box(path: Path) -> None:
    vertices = (
        (-1.6, -1.0, -0.7, 0.88, 0.22, 0.28),
        (1.6, -1.0, -0.7, 0.22, 0.76, 0.42),
        (1.6, 1.0, -0.7, 0.22, 0.48, 0.92),
        (-1.6, 1.0, -0.7, 0.92, 0.72, 0.20),
        (-1.6, -1.0, 0.7, 0.78, 0.28, 0.88),
        (1.6, -1.0, 0.7, 0.20, 0.82, 0.88),
        (1.6, 1.0, 0.7, 0.94, 0.94, 0.94),
        (-1.6, 1.0, 0.7, 0.36, 0.56, 0.96),
    )
    faces = (
        (1, 2, 3), (1, 3, 4),
        (5, 7, 6), (5, 8, 7),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 8), (3, 8, 4),
        (4, 8, 5), (4, 5, 1),
    )
    lines = ["# Phase 9.3C synthetic canonical scene"]
    lines.extend("v " + " ".join(str(value) for value in vertex) for vertex in vertices)
    lines.extend("f " + " ".join(str(value) for value in face) for face in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _camera_poses() -> np.ndarray:
    # Presentation-only synthetic path in the same reconstruction coordinate frame
    # as the box. Identity rotations keep the pose construction unambiguous while
    # translations make the trajectory/frustums easy to inspect visually.
    positions = (
        (-2.2, 0.35, -1.6),
        (-1.5, 0.15, -1.25),
        (-0.8, 0.0, -0.95),
        (0.0, -0.10, -0.70),
        (0.8, -0.05, -0.45),
        (1.5, 0.10, -0.15),
        (2.1, 0.30, 0.20),
    )
    matrices = []
    for position in positions:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = np.asarray(position, dtype=np.float64)
        matrices.append(matrix)
    return np.asarray(matrices, dtype=np.float64)


def main() -> int:
    args = _parser().parse_args()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    with tempfile.TemporaryDirectory(prefix="phase9_3c_preview_") as tmp:
        root = Path(tmp).resolve()
        _fixture, _providers, result, snapshots = run_f5_fixture(
            root / "run", request_id="phase9-3c-preview"
        )
        base = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
        camera_artifact = next(
            item for item in base.artifacts if item.kind is ReconstructionArtifactKind.CAMERA_TRAJECTORY
        )
        surface = (root / "canonical_surface.obj").resolve()
        _write_colored_box(surface)
        poses = _camera_poses()
        snapshots = dict(snapshots)
        snapshots["surface_mesh"] = {
            "surface_obj_path": str(surface),
            "surface_vertex_count": 8,
            "surface_face_count": 12,
            "metric_geometry_status": "arbitrary",
            "canonical_geometry_authoritative": True,
            "certified_metrology": False,
        }
        snapshots["camera_trajectory"] = build_canonical_camera_trajectory_snapshot(
            matrices_reconstruction=poses,
            camera_ids=tuple(f"camera-{index:02d}" for index in range(poses.shape[0])),
            pose_layers=("optimized",) * poses.shape[0],
            provider_id=camera_artifact.provider_id,
            reconstruction_frame_id="reconstruction-world",
        )
        scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)

        page = ReconstructionPage()
        page.resize(1366, 768)
        page.set_persistent_scene(scene, "phase9_3c_preview/persistent_reconstruction_scene.json")
        page._canonical_scene_group.set_expanded(True)
        page.show()
        for _ in range(12):
            app.processEvents()
            time.sleep(0.02)

        canvas = page._reconstruction_canvas
        if page.canonical_viewer_mesh is None or not canvas.canonical_mesh_loaded:
            raise RuntimeError("Phase 9.3C preview did not load the persistent canonical surface.")
        if page.canonical_camera_presentation is None or not canvas.canonical_camera_loaded:
            raise RuntimeError("Phase 9.3C preview did not load persistent canonical camera context.")
        if not canvas.canonical_camera_frustums_visible or not canvas.canonical_camera_trajectory_visible:
            raise RuntimeError("Phase 9.3C camera frustums/trajectory are unexpectedly hidden.")

        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if not page.grab().save(str(output), "PNG"):
            raise RuntimeError(f"Unable to save preview screenshot: {output}")
        print(f"Phase 9.3C real-VisPy world/camera preview: {output}")
        print(
            "canonical_vertices=8 canonical_faces=12 persisted_camera_poses=7 "
            "camera_authority=supporting_geometry certified_metrology=False"
        )
        print("No reconstruction/DA3/Gaussian/NeRF backend was launched.")

        if args.show:
            exit_code = app.exec()
            page.shutdown_viewport()
            if canvas.canonical_mesh_loaded or canvas.canonical_camera_loaded:
                raise RuntimeError("Canonical viewer resources remained loaded after viewport shutdown.")
            page.close()
            print("Canonical mesh/camera viewer resources released on viewport shutdown.")
            return int(exit_code)
        page.shutdown_viewport()
        if canvas.canonical_mesh_loaded or canvas.canonical_camera_loaded:
            raise RuntimeError("Canonical viewer resources remained loaded after viewport shutdown.")
        page.close()
        print("Canonical mesh/camera viewer resources released on viewport shutdown.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
