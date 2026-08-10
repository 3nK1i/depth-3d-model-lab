#!/usr/bin/env python3
"""Real PyCOLMAP -> D1 bounded artifact -> VisPy offline visualization smoke."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.reconstruction_persistence_controller import ReconstructionSceneAvailability
from app.widgets.mode_tabs import AppMode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument(
        "--sfm-python",
        type=Path,
        default=Path.home() / "anaconda3/envs/eia-pycolmap-probe/bin/python",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/phase8_8d1_offline_visualization"),
    )
    parser.add_argument("--timeout-minutes", type=float, default=15.0)
    args = parser.parse_args()

    images = args.images.expanduser().resolve()
    sfm_python = args.sfm_python.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not images.is_dir():
        raise RuntimeError(f"Image sequence does not exist: {images}")
    if not sfm_python.is_file():
        raise RuntimeError(f"PyCOLMAP interpreter does not exist: {sfm_python}")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    runtime_root = output / "runs"

    os.environ["EIA_RECONSTRUCTION_PYCOLMAP_PYTHON"] = str(sfm_python)
    os.environ["EIA_RECONSTRUCTION_OUTPUT_ROOT"] = str(runtime_root)
    # D1 is a real rendering gate: deliberately do not force Qt offscreen/fake VisPy.
    os.environ.pop("DEPTH3D_TEST_FAKE_VISPY", None)

    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = MainWindow()
    window.show()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    app.processEvents()
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    try:
        selection = controller.select_image_sequence(images)
        app.processEvents()
        if not selection.ready:
            raise RuntimeError(f"D1 source did not become ready: {selection.detail}")
        page._start_button.click()
        app.processEvents()
        session_id = controller.offline_session_id
        if not session_id:
            raise RuntimeError("D1 did not create an offline session.")

        terminal = {
            OfflineReconstructionRunState.COMPLETED,
            OfflineReconstructionRunState.FAILED,
            OfflineReconstructionRunState.CANCELLED,
        }
        deadline = monotonic() + float(args.timeout_minutes) * 60.0
        while controller.offline_state not in terminal and monotonic() < deadline:
            app.processEvents()
            sleep(0.05)
        for _ in range(10):
            app.processEvents()
            sleep(0.02)
        if controller.offline_state not in terminal:
            controller.cancel_offline()
            raise RuntimeError("D1 offline visualization smoke timed out.")

        run_dir = controller.offline_output_dir
        result_path = (run_dir or Path("/missing")) / "offline_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        artifact_text = str(result.get("artifacts", {}).get("offline_visualization_json", ""))
        artifact = Path(artifact_text) if artifact_text else Path("/missing")
        canvas = page._reconstruction_canvas
        screenshot = output / "reconstruction_visualizer.png"
        app.processEvents()
        window.grab().save(str(screenshot))

        registered = int(result.get("sfm_summary", {}).get("registered_keyframes", 0) or 0)
        landmarks = int(result.get("sfm_summary", {}).get("landmarks", 0) or 0)
        scale_status = str(result.get("sfm_summary", {}).get("gauge", {}).get("scale_status", ""))
        passed = bool(
            controller.offline_state is OfflineReconstructionRunState.COMPLETED
            and result.get("pass") is True
            and artifact.is_file()
            and canvas.offline_visualization_loaded
            and canvas.offline_session_id == session_id
            and canvas.offline_scale_status == "arbitrary"
            and canvas.sparse_landmark_count == landmarks
            and 0 < canvas.sparse_landmark_display_count <= canvas.MAX_SPARSE_LANDMARK_DISPLAY_POINTS
            and canvas.keyframe_count == registered
            and canvas.keyframe_display_count > 0
            and canvas.trajectory_count == 0
            and canvas.has_pose is False
            and canvas.sparse_landmarks_visible
            and canvas.keyframes_visible
            and page._viewport_placeholder.isHidden()
            and page._viewport_state.text() == "Offline sparse reconstruction"
            and controller.scene_availability is ReconstructionSceneAvailability.PARTIAL
            and scale_status == "arbitrary"
            and result.get("canonical_geometry_authoritative") is False
            and result.get("certified_metrology") is False
            and result.get("execution_boundary", {}).get("pycolmap_external_process") is True
            and result.get("execution_boundary", {}).get("production_depth_imports_pycolmap") is False
            and "pycolmap" not in sys.modules
            and not errors
        )
        summary = {
            "phase": "8.8D1",
            "pass": passed,
            "session_id": session_id,
            "controller_state": controller.offline_state.value,
            "scene_availability": controller.scene_availability.value,
            "offline_visualization_artifact": str(artifact),
            "offline_visualization_loaded": canvas.offline_visualization_loaded,
            "offline_visualization_session_id": canvas.offline_session_id,
            "offline_scale_status": canvas.offline_scale_status,
            "registered_keyframes": registered,
            "camera_display_count": canvas.keyframe_display_count,
            "landmarks": landmarks,
            "landmark_display_count": canvas.sparse_landmark_display_count,
            "trajectory_count": canvas.trajectory_count,
            "canvas_has_live_pose": canvas.has_pose,
            "sparse_landmarks_visible": canvas.sparse_landmarks_visible,
            "registered_cameras_visible": canvas.keyframes_visible,
            "viewport_state": page._viewport_state.text(),
            "viewport_placeholder_hidden": page._viewport_placeholder.isHidden(),
            "scale_status": scale_status,
            "canonical_geometry_authoritative": result.get("canonical_geometry_authoritative"),
            "certified_metrology": result.get("certified_metrology"),
            "pycolmap_external_process": result.get("execution_boundary", {}).get("pycolmap_external_process"),
            "production_depth_imports_pycolmap": result.get("execution_boundary", {}).get("production_depth_imports_pycolmap"),
            "pycolmap_in_application_process": "pycolmap" in sys.modules,
            "screenshot": str(screenshot),
            "errors": errors,
        }
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Phase 8.8D1 offline visualization smoke: {'PASS' if passed else 'FAIL'}")
        print(
            f"session_id={session_id} registered={registered} cameras_shown={canvas.keyframe_display_count} "
            f"landmarks={landmarks} points_shown={canvas.sparse_landmark_display_count}"
        )
        print(f"viewport={page._viewport_state.text()!r} scene={controller.scene_availability.value}")
        print(f"screenshot={screenshot}")
        print(f"summary={summary_path}")
        return 0 if passed else 1
    finally:
        window.close()
        app.processEvents()


if __name__ == "__main__":
    raise SystemExit(main())
