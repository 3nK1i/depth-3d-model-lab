#!/usr/bin/env python3
"""Real South Building PyCOLMAP -> D2 camera inspection -> VisPy smoke."""

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
        default=Path("/tmp/phase8_8d2_offline_inspection"),
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

    os.environ["EIA_RECONSTRUCTION_PYCOLMAP_PYTHON"] = str(sfm_python)
    os.environ["EIA_RECONSTRUCTION_OUTPUT_ROOT"] = str(output / "runs")
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
            raise RuntimeError(f"D2 source did not become ready: {selection.detail}")
        page._start_button.click()
        app.processEvents()
        session_id = controller.offline_session_id
        if not session_id:
            raise RuntimeError("D2 did not create an offline session.")

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
            raise RuntimeError("D2 offline inspection smoke timed out.")

        run_dir = controller.offline_output_dir
        result_path = (run_dir or Path("/missing")) / "offline_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        canvas = page._reconstruction_canvas
        visual = page._offline_visualization
        if visual is None or not visual.camera_ids:
            raise RuntimeError("D2 did not load inspectable registered cameras.")
        target = len(visual.camera_ids) // 2
        page._offline_camera_combo.setCurrentIndex(target + 1)
        app.processEvents()
        page._offline_focus_camera_button.click()
        app.processEvents()

        screenshot = output / "reconstruction_visualizer.png"
        window.grab().save(str(screenshot))
        registered = int(result.get("sfm_summary", {}).get("registered_keyframes", 0) or 0)
        landmarks = int(result.get("sfm_summary", {}).get("landmarks", 0) or 0)
        selected_id = visual.camera_ids[target]
        passed = bool(
            controller.offline_state is OfflineReconstructionRunState.COMPLETED
            and result.get("pass") is True
            and canvas.offline_visualization_loaded
            and canvas.offline_registered_camera_count == len(visual.camera_ids)
            and canvas.offline_camera_path_visible
            and 1 < canvas.offline_camera_path_display_count <= canvas.MAX_OFFLINE_CAMERA_PATH_POINTS
            and canvas.selected_offline_camera_index == target
            and canvas.selected_offline_camera_id == selected_id
            and canvas.selected_offline_camera_visible
            and "gauge" in page._offline_camera_position_value.text()
            and page._offline_focus_camera_button.isEnabled()
            and canvas.sparse_landmark_count == landmarks
            and canvas.keyframe_count == registered
            and canvas.trajectory_count == 0
            and canvas.has_pose is False
            and controller.scene_availability is ReconstructionSceneAvailability.PARTIAL
            and canvas.offline_scale_status == "arbitrary"
            and result.get("canonical_geometry_authoritative") is False
            and result.get("certified_metrology") is False
            and result.get("execution_boundary", {}).get("pycolmap_external_process") is True
            and result.get("execution_boundary", {}).get("production_depth_imports_pycolmap") is False
            and "pycolmap" not in sys.modules
            and not errors
        )
        summary = {
            "phase": "8.8D2",
            "pass": passed,
            "session_id": session_id,
            "controller_state": controller.offline_state.value,
            "scene_availability": controller.scene_availability.value,
            "registered_keyframes": registered,
            "offline_registered_camera_count": canvas.offline_registered_camera_count,
            "offline_camera_path_visible": canvas.offline_camera_path_visible,
            "offline_camera_path_display_count": canvas.offline_camera_path_display_count,
            "selected_camera_index": canvas.selected_offline_camera_index,
            "selected_camera_id": canvas.selected_offline_camera_id,
            "selected_camera_visible": canvas.selected_offline_camera_visible,
            "selected_camera_position_ui": page._offline_camera_position_value.text(),
            "selected_camera_delta_ui": page._offline_camera_delta_value.text(),
            "offline_structure_ui": page._offline_structure_value.text(),
            "landmarks": landmarks,
            "landmark_display_count": canvas.sparse_landmark_display_count,
            "trajectory_count": canvas.trajectory_count,
            "canvas_has_live_pose": canvas.has_pose,
            "offline_scale_status": canvas.offline_scale_status,
            "canonical_geometry_authoritative": result.get("canonical_geometry_authoritative"),
            "certified_metrology": result.get("certified_metrology"),
            "pycolmap_external_process": result.get("execution_boundary", {}).get("pycolmap_external_process"),
            "production_depth_imports_pycolmap": result.get("execution_boundary", {}).get("production_depth_imports_pycolmap"),
            "pycolmap_in_application_process": "pycolmap" in sys.modules,
            "viewport_state": page._viewport_state.text(),
            "screenshot": str(screenshot),
            "errors": errors,
        }
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Phase 8.8D2 offline inspection smoke: {'PASS' if passed else 'FAIL'}")
        print(
            f"session_id={session_id} registered={registered} selected={selected_id!r} "
            f"landmarks={landmarks} points_shown={canvas.sparse_landmark_display_count}"
        )
        print(f"position={page._offline_camera_position_value.text()!r}")
        print(f"screenshot={screenshot}")
        print(f"summary={summary_path}")
        return 0 if passed else 1
    finally:
        window.close()
        app.processEvents()


if __name__ == "__main__":
    raise SystemExit(main())
