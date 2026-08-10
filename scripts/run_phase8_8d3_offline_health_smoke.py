#!/usr/bin/env python3
"""Real South Building PyCOLMAP -> D3 sparse-SfM health -> Qt/VisPy smoke."""

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
        default=Path("/tmp/phase8_8d3_offline_health"),
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
            raise RuntimeError(f"D3 source did not become ready: {selection.detail}")
        page._start_button.click()
        app.processEvents()
        session_id = controller.offline_session_id
        if not session_id:
            raise RuntimeError("D3 did not create an offline session.")

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
            raise RuntimeError("D3 offline health smoke timed out.")

        run_dir = controller.offline_output_dir
        result_path = (run_dir or Path("/missing")) / "offline_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        sfm = result.get("sfm_summary", {}) if isinstance(result, dict) else {}
        quality = sfm.get("quality_assessment", {}) if isinstance(sfm, dict) else {}
        canvas = page._reconstruction_canvas
        visual = page._offline_visualization
        health = page._offline_sfm_health
        if visual is None or health is None:
            raise RuntimeError("D3 did not load offline visualization and SfM health presentation.")

        target = len(visual.camera_ids) // 2
        page._offline_camera_combo.setCurrentIndex(target + 1)
        app.processEvents()
        screenshot = output / "reconstruction_visualizer.png"
        window.grab().save(str(screenshot))

        quality_status = str(quality.get("status", "")).strip().lower() if isinstance(quality, dict) else ""
        quality_usable = quality.get("usable") if isinstance(quality, dict) else None
        registered = int(sfm.get("registered_keyframes", 0) or 0) if isinstance(sfm, dict) else 0
        input_keyframes = int(sfm.get("input_keyframes", 0) or 0) if isinstance(sfm, dict) else 0
        landmarks = int(sfm.get("landmarks", 0) or 0) if isinstance(sfm, dict) else 0
        observations = int(sfm.get("observations", 0) or 0) if isinstance(sfm, dict) else 0

        passed = bool(
            controller.offline_state is OfflineReconstructionRunState.COMPLETED
            and result.get("pass") is True
            and health.authoritative_assessment
            and health.status == quality_status
            and health.usable is quality_usable
            and quality_status in {"accepted", "degraded"}
            and page._offline_quality_status_value.text() == quality_status.upper()
            and "/" in page._offline_quality_registration_value.text()
            and "%" in page._offline_quality_registration_value.text()
            and "px mean" in page._offline_quality_reprojection_value.text()
            and "views / point" in page._offline_quality_support_value.text()
            and "points / camera" in page._offline_quality_support_value.text()
            and "metrology" not in page._offline_quality_status_value.text().lower()
            and canvas.offline_visualization_loaded
            and canvas.offline_registered_camera_count == len(visual.camera_ids)
            and canvas.selected_offline_camera_index == target
            and canvas.selected_offline_camera_visible
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
            and registered > 0
            and input_keyframes >= registered
            and landmarks > 0
            and observations >= landmarks * 2
            and not errors
        )
        summary = {
            "phase": "8.8D3",
            "pass": passed,
            "session_id": session_id,
            "controller_state": controller.offline_state.value,
            "scene_availability": controller.scene_availability.value,
            "quality_status": health.status,
            "quality_usable": health.usable,
            "quality_authoritative_assessment": health.authoritative_assessment,
            "quality_registration_ui": page._offline_quality_registration_value.text(),
            "quality_reprojection_ui": page._offline_quality_reprojection_value.text(),
            "quality_support_ui": page._offline_quality_support_value.text(),
            "quality_gate_ui": page._offline_quality_gate_value.text(),
            "quality_reasons": list(health.reasons),
            "registered_keyframes": registered,
            "input_keyframes": input_keyframes,
            "landmarks": landmarks,
            "observations": observations,
            "landmark_display_count": canvas.sparse_landmark_display_count,
            "selected_camera_index": canvas.selected_offline_camera_index,
            "selected_camera_id": canvas.selected_offline_camera_id,
            "selected_camera_visible": canvas.selected_offline_camera_visible,
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
        print(f"Phase 8.8D3 offline SfM health smoke: {'PASS' if passed else 'FAIL'}")
        print(
            f"session_id={session_id} quality={health.status!r} registration={registered}/{input_keyframes} "
            f"landmarks={landmarks} observations={observations}"
        )
        print(f"health={page._offline_quality_support_value.text()!r}")
        print(f"screenshot={screenshot}")
        print(f"summary={summary_path}")
        return 0 if passed else 1
    finally:
        window.close()
        app.processEvents()


if __name__ == "__main__":
    raise SystemExit(main())
