#!/usr/bin/env python3
"""Phase 8.8H5 real Reconstruction UI acceptance with the isolated PyCOLMAP backend."""

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

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.reconstruction_controller import ReconstructionWorkflow
from app.reconstruction_persistence_controller import ReconstructionSceneAvailability
from app.widgets.mode_tabs import AppMode


def _pump(app: QApplication, seconds: float = 0.05) -> None:
    deadline = monotonic() + max(0.0, float(seconds))
    while monotonic() < deadline:
        app.processEvents()
        sleep(0.01)
    app.processEvents()


def _save_window(window: MainWindow, path: Path) -> None:
    pixmap = window.grab()
    if pixmap.isNull() or not pixmap.save(str(path)):
        raise RuntimeError(f"Could not save H5 application screenshot: {path}")


def _sidebar_metrics(page) -> dict[str, object]:
    scroll = page._settings_scroll
    body = scroll.widget()
    hbar = scroll.horizontalScrollBar()
    return {
        "scroll_width": scroll.width(),
        "viewport_width": scroll.viewport().width(),
        "body_width": 0 if body is None else body.width(),
        "horizontal_maximum": hbar.maximum(),
        "horizontal_visible": hbar.isVisible(),
        "horizontal_policy_always_off": (
            scroll.horizontalScrollBarPolicy() is Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        ),
    }


def _write_summary(output: Path, summary: dict[str, object]) -> Path:
    path = output / "summary.json"
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path, help="Top-level directory of overlapping images.")
    source.add_argument("--video", type=Path, help="Recorded video for offline reconstruction.")
    parser.add_argument(
        "--sfm-python",
        type=Path,
        default=Path.home() / "anaconda3/envs/eia-pycolmap-probe/bin/python",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/phase8_8h5_real_ui_acceptance"),
    )
    parser.add_argument("--timeout-minutes", type=float, default=20.0)
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.0,
        help="Keep the completed real UI visible briefly before closing.",
    )
    args = parser.parse_args()

    sfm_python = args.sfm_python.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not sfm_python.is_file():
        raise RuntimeError(f"PyCOLMAP interpreter does not exist: {sfm_python}")
    if args.timeout_minutes <= 0.0:
        parser.error("--timeout-minutes must be positive.")
    if args.hold_seconds < 0.0:
        parser.error("--hold-seconds cannot be negative.")

    images = args.images.expanduser().resolve() if args.images is not None else None
    video = args.video.expanduser().resolve() if args.video is not None else None
    if images is not None and not images.is_dir():
        raise RuntimeError(f"Image sequence does not exist: {images}")
    if video is not None and not video.is_file():
        raise RuntimeError(f"Recorded video does not exist: {video}")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    os.environ["EIA_RECONSTRUCTION_PYCOLMAP_PYTHON"] = str(sfm_python)
    os.environ["EIA_RECONSTRUCTION_OUTPUT_ROOT"] = str(output / "runs")
    # H5 is a real viewport/UI acceptance. Never inherit fake/offscreen VisPy flags.
    os.environ.pop("DEPTH3D_TEST_FAKE_VISPY", None)
    os.environ.pop("QT_QPA_PLATFORM", None)
    os.environ.pop("QT_OPENGL", None)
    os.environ.pop("LIBGL_ALWAYS_SOFTWARE", None)

    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = MainWindow()
    window.resize(1366, 768)
    window.show()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    _pump(app, 0.15)

    page = window._reconstruction_page
    controller = window._reconstruction_controller
    errors: list[str] = []
    events: list[dict[str, object]] = []
    controller.error_occurred.connect(errors.append)
    controller.offline_event_received.connect(
        lambda payload: events.append(dict(payload)) if isinstance(payload, dict) else None
    )

    source_ready_png = output / "01_source_ready.png"
    active_png = output / "02_runtime_active.png"
    completed_png = output / "03_completed_inspection.png"
    active_captured = False
    summary: dict[str, object] = {
        "phase": "8.8H5",
        "pass": False,
        "real_application_executed": True,
        "real_external_pycolmap_executed": False,
        "heavy_gaussian_or_nerf_executed_by_h5": False,
        "hardware_profile_claimed_by_h5": False,
        "production_behavior_changed_by_h5": False,
        "new_feature_work_added_by_h5": False,
        "release_tag_created_by_h5": False,
        "certified_metrology": False,
        "errors": errors,
    }

    try:
        if images is not None:
            selection = controller.select_image_sequence(images)
            expected_workflow = ReconstructionWorkflow.IMAGE_SEQUENCE
        else:
            selection = controller.select_recorded_video(video)
            expected_workflow = ReconstructionWorkflow.RECORDED_VIDEO
        _pump(app, 0.1)

        source_ready = bool(
            selection.ready
            and selection.workflow is expected_workflow
            and page._start_button.isEnabled()
            and page._source_readiness_value.text() == "Ready"
        )
        _save_window(window, source_ready_png)

        page._start_button.click()
        _pump(app, 0.05)
        session_id = controller.offline_session_id or ""
        terminal = {
            OfflineReconstructionRunState.COMPLETED,
            OfflineReconstructionRunState.FAILED,
            OfflineReconstructionRunState.CANCELLED,
        }
        deadline = monotonic() + float(args.timeout_minutes) * 60.0
        while controller.offline_state not in terminal and monotonic() < deadline:
            _pump(app, 0.05)
            if (
                not active_captured
                and controller.offline_state is OfflineReconstructionRunState.RUNNING
                and (page._progress_bar.value() >= 1 or len(events) >= 2)
            ):
                _save_window(window, active_png)
                active_captured = True
        _pump(app, 0.25)
        if controller.offline_state not in terminal:
            controller.cancel_offline()
            raise RuntimeError("H5 real offline reconstruction timed out.")

        run_dir = controller.offline_output_dir
        result_path = (run_dir or Path("/missing")) / "offline_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        visual = page._offline_visualization
        canvas = page._reconstruction_canvas

        if visual is not None and visual.camera_ids:
            target = len(visual.camera_ids) // 2
            page._offline_camera_combo.setCurrentIndex(target + 1)
            _pump(app, 0.05)
            page._offline_focus_camera_button.click()
            _pump(app, 0.08)
        else:
            target = -1

        # Open the key real-acceptance groups before the final screenshot. This is
        # presentation-only and deliberately exercises the 1366x768 disclosure stack.
        for group in (
            page._workflow_group,
            page._session_group,
            page._offline_inspection_group,
            page._appearance_group,
            page._authority_group,
        ):
            group.set_expanded(True)
        _pump(app, 0.1)
        _save_window(window, completed_png)

        sfm = result.get("sfm_summary", {}) if isinstance(result, dict) else {}
        registered = int(sfm.get("registered_keyframes", 0) or 0) if isinstance(sfm, dict) else 0
        landmarks = int(sfm.get("landmarks", 0) or 0) if isinstance(sfm, dict) else 0
        sidebar = _sidebar_metrics(page)
        selected_ok = bool(
            target >= 0
            and visual is not None
            and canvas.selected_offline_camera_index == target
            and canvas.selected_offline_camera_visible
            and page._offline_focus_camera_button.isEnabled()
        )
        appearance_truthful = bool(
            not page.gaussian_preview_visible
            and not page.nerf_preview_visible
        )
        responsive_ok = bool(
            window.width() >= 1366
            and window.height() >= 768
            and sidebar["horizontal_maximum"] == 0
            and sidebar["horizontal_visible"] is False
            and sidebar["horizontal_policy_always_off"] is True
        )
        resource_release_ok = bool(
            page._start_button.isEnabled()
            and not page._pause_button.isEnabled()
            and not page._stop_button.isEnabled()
            and not controller.is_active
        )
        real_boundary_ok = bool(
            result.get("production_input_assembly") is True
            and result.get("tests_fixture_used") is False
            and result.get("execution_boundary", {}).get("pycolmap_external_process") is True
            and result.get("execution_boundary", {}).get("production_depth_imports_pycolmap") is False
            and "pycolmap" not in sys.modules
        )
        real_visual_ok = bool(
            controller.offline_state is OfflineReconstructionRunState.COMPLETED
            and result.get("pass") is True
            and visual is not None
            and canvas.offline_visualization_loaded
            and registered >= 2
            and landmarks > 0
            and canvas.offline_registered_camera_count == len(visual.camera_ids)
            and canvas.keyframe_count == registered
            and canvas.sparse_landmark_count == landmarks
            and page._viewport_placeholder.isHidden()
        )
        authority_ok = bool(
            controller.scene_availability is ReconstructionSceneAvailability.PARTIAL
            and result.get("canonical_geometry_authoritative") is False
            and result.get("certified_metrology") is False
            and "Not certified" in page._metrology_value.text()
        )
        passed = bool(
            source_ready
            and session_id
            and real_visual_ok
            and selected_ok
            and responsive_ok
            and resource_release_ok
            and real_boundary_ok
            and authority_ok
            and appearance_truthful
            and not errors
        )
        summary.update(
            {
                "pass": passed,
                "source_ready": source_ready,
                "source_workflow": selection.workflow.value,
                "source_path": str(selection.path or ""),
                "session_id": session_id,
                "controller_state": controller.offline_state.value,
                "scene_availability": controller.scene_availability.value,
                "registered_keyframes": registered,
                "landmarks": landmarks,
                "offline_visualization_loaded": canvas.offline_visualization_loaded,
                "selected_camera_index": canvas.selected_offline_camera_index,
                "selected_camera_id": canvas.selected_offline_camera_id,
                "selected_camera_visible": canvas.selected_offline_camera_visible,
                "viewport_state": page._viewport_state.text(),
                "real_external_pycolmap_executed": bool(real_boundary_ok),
                "production_input_assembly": result.get("production_input_assembly"),
                "tests_fixture_used": result.get("tests_fixture_used"),
                "pycolmap_external_process": result.get("execution_boundary", {}).get("pycolmap_external_process"),
                "production_depth_imports_pycolmap": result.get("execution_boundary", {}).get("production_depth_imports_pycolmap"),
                "pycolmap_in_application_process": "pycolmap" in sys.modules,
                "canonical_geometry_authoritative": result.get("canonical_geometry_authoritative"),
                "certified_metrology": result.get("certified_metrology"),
                "appearance_preview_truthful_without_artifacts": appearance_truthful,
                "resource_release_after_completion": resource_release_ok,
                "responsive_1366x768": responsive_ok,
                "sidebar": sidebar,
                "event_count": len(events),
                "source_ready_screenshot": str(source_ready_png),
                "runtime_active_screenshot": str(active_png) if active_captured else "",
                "completed_screenshot": str(completed_png),
                "offline_result": str(result_path),
                "errors": errors,
            }
        )
        summary_path = _write_summary(output, summary)
        print(f"Phase 8.8H5 real Reconstruction UI acceptance: {'PASS' if passed else 'FAIL'}")
        print(
            f"workflow={selection.workflow.value} state={controller.offline_state.value} "
            f"registered={registered} landmarks={landmarks}"
        )
        print(
            "responsive_1366x768="
            f"{responsive_ok} resources_released={resource_release_ok} "
            f"external_pycolmap={real_boundary_ok}"
        )
        print(f"completed_screenshot={completed_png}")
        print(f"summary={summary_path}")
        if args.hold_seconds > 0.0:
            _pump(app, args.hold_seconds)
        return 0 if passed else 1
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        summary["errors"] = errors
        summary_path = _write_summary(output, summary)
        print(f"Phase 8.8H5 real Reconstruction UI acceptance: FAIL ({exc})")
        print(f"summary={summary_path}")
        return 1
    finally:
        window.close()
        _pump(app, 0.05)


if __name__ == "__main__":
    raise SystemExit(main())
