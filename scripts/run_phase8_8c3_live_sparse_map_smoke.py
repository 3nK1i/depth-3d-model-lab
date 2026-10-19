#!/usr/bin/env python3
"""Real Phase 8.8C3 MainWindow -> Stella sparse map -> VisPy smoke.

This smoke intentionally requires a normal desktop/display session. Unlike the
B3/B4 headless controller/status smokes, it uses the actual VisPy reconstruction
canvas and therefore validates a real OpenGL-backed presentation surface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from time import monotonic, sleep

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.live_slam_controller import LiveStellaLaunchSettings
from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from core.capture.base_source import SourceStatus
from core.capture.video_source import VideoSource
from core.reconstruction import CameraIntrinsicsPolicy, LiveCaptureRuntimeState
from scripts.run_live_capture_stella_smoke import _build_euroc_video
from scripts.run_live_stella_worker_smoke import _load_euroc_index


def _wait_until(app: QApplication, predicate, *, timeout_s: float) -> bool:
    deadline = monotonic() + max(0.0, float(timeout_s))
    while monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        sleep(0.005)
    app.processEvents()
    return bool(predicate())


def _require(path: Path, *, directory: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    ok = resolved.is_dir() if directory else resolved.is_file()
    if not ok:
        raise RuntimeError(f"Required path is missing: {resolved}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stella-prefix", type=Path, default=Path.home() / "Desktop/depth3d-stella-vslam-probe")
    parser.add_argument("--worker", type=Path, default=Path.home() / "Desktop/depth3d-stella-live-worker/bin/depth3d_stella_live_worker")
    parser.add_argument("--vocab", type=Path, default=Path.home() / "Desktop/phase8_4_stella_probe_data/orb_vocab.fbow")
    parser.add_argument("--euroc", type=Path, default=Path.home() / "Desktop/phase8_4_stella_probe_data/MH_01_easy/mav0")
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--playback-speed", type=float, default=2.0)
    parser.add_argument("--observe", type=int, default=70)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/phase8_8c3_live_sparse_map"))
    args = parser.parse_args()

    stella_prefix = _require(args.stella_prefix, directory=True)
    worker = _require(args.worker)
    vocab = _require(args.vocab)
    euroc = _require(args.euroc, directory=True)
    records = _load_euroc_index(euroc, args.max_frames)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    window.showMaximized()
    app.processEvents()

    page = window._reconstruction_page
    controller = window._reconstruction_controller
    application_controller = window._controller
    statuses: list[SourceStatus] = []
    observations: list[object] = []
    errors: list[str] = []
    application_controller.source_status_changed.connect(statuses.append)
    controller.live_observation_ready.connect(observations.append)
    controller.error_occurred.connect(errors.append)

    session_id: str | None = None
    cleanup_ok = False
    retained_after_stop = False
    screenshot_path = output_dir / "reconstruction_visualizer.png"
    try:
        with tempfile.TemporaryDirectory(prefix="depth3d-phase8-c3-live-sparse-map-") as temp_text:
            video_path = Path(temp_text) / "euroc_cam0.avi"
            _build_euroc_video(records, video_path)
            video = VideoSource(str(video_path))
            video.set_loop(False)
            video.set_speed(args.playback_speed)
            application_controller._replace_active_source(video)
            if not _wait_until(
                app,
                lambda: SourceStatus.CONNECTED in statuses and application_controller.latest_source_frame is not None,
                timeout_s=8.0,
            ):
                raise RuntimeError("Simulated-live source did not provide a frame.")

            selection = controller.select_live_source()
            if not selection.ready:
                raise RuntimeError(f"Live selection is not ready: {selection.detail}")
            camera = controller.build_estimated_live_camera()
            page._live_camera = camera
            page._live_camera_mode = "estimated"
            page._configure_visualizer_camera(camera)
            page._refresh_live_camera_presentation()

            session_id = controller.start_selected_live(
                camera=camera,
                settings=LiveStellaLaunchSettings(
                    install_prefix=stella_prefix,
                    worker_executable=worker,
                    vocabulary_path=vocab,
                    startup_timeout_s=20.0,
                    frame_timeout_s=5.0,
                    shutdown_timeout_s=5.0,
                    require_qualified_build=True,
                ),
                camera_intrinsics_policy=CameraIntrinsicsPolicy.ALLOW_ESTIMATED,
            )
            if not _wait_until(
                app,
                lambda: controller.live_state in {LiveCaptureRuntimeState.RUNNING, LiveCaptureRuntimeState.FAILED},
                timeout_s=25.0,
            ):
                raise RuntimeError("C3 sparse-map smoke did not reach RUNNING.")
            if controller.live_state is LiveCaptureRuntimeState.FAILED:
                raise RuntimeError(f"C3 live startup failed: {errors}")
            if not _wait_until(app, lambda: len(observations) >= args.observe, timeout_s=40.0):
                raise RuntimeError(f"Insufficient live observations: {len(observations)}; errors={errors}")
            if not _wait_until(app, lambda: page._reconstruction_canvas.has_pose, timeout_s=3.0):
                raise RuntimeError("The C3 VisPy canvas never received a live camera pose.")
            if not _wait_until(
                app,
                lambda: page._reconstruction_canvas.sparse_landmark_display_count >= 20,
                timeout_s=8.0,
            ):
                raise RuntimeError(
                    "The C3 worker did not deliver a usable sparse landmark sample. "
                    "Rebuild it with scripts/build_stella_live_worker.sh before rerunning this smoke."
                )

            canvas = page._reconstruction_canvas
            position = canvas.current_camera_position
            if position is None or not np.isfinite(position).all():
                raise RuntimeError("C3 current-camera viewer position is unavailable or non-finite.")
            window.grab().save(str(screenshot_path))

            if not controller.request_stop_live():
                raise RuntimeError("C3 live stop request was rejected.")
            if not _wait_until(
                app,
                lambda: controller.live_state in {LiveCaptureRuntimeState.STOPPED, LiveCaptureRuntimeState.FAILED},
                timeout_s=12.0,
            ):
                raise RuntimeError("C3 sparse-map smoke did not stop cleanly.")
            # Validate the terminal runtime + UI state *before* disposing the
            # stopped runtime. LiveSlamCaptureController.stop() clears its runtime
            # on successful cleanup, after which controller.live_state correctly
            # reports IDLE rather than STOPPED.
            terminal_ui_ready = _wait_until(
                app,
                lambda: (
                    controller.live_state is LiveCaptureRuntimeState.STOPPED
                    and page._live_state is LiveCaptureRuntimeState.STOPPED
                    and page._viewport_state.text() == "Live camera pose retained"
                ),
                timeout_s=2.0,
            )
            retained_after_stop = bool(terminal_ui_ready and canvas.has_pose)
            cleanup_ok = controller.stop_live(timeout_s=2.0)
            application_controller.disconnect_active_source()
            app.processEvents()

        canvas = page._reconstruction_canvas
        position = canvas.current_camera_position
        trajectory_display_count = canvas.trajectory_display_count
        sparse_display_count = canvas.sparse_landmark_display_count
        sparse_total_count = canvas.sparse_landmark_count
        sparse_map_revision = canvas.sparse_map_revision
        world_axis_length = canvas.world_axis_length
        orientation_projection = canvas.orientation_gizmo_axes()
        orientation_gizmo_ready = bool(
            not page._orientation_gizmo.isHidden()
            and orientation_projection.shape == (3, 2)
            and np.isfinite(orientation_projection).all()
        )
        world_axis_arrowheads = all(
            hasattr(canvas, name) for name in ("_axis_x_head", "_axis_y_head", "_axis_z_head")
        )
        passed = bool(
            session_id
            and not errors
            and cleanup_ok
            and canvas.has_pose
            and canvas.trajectory_count > 1
            and trajectory_display_count > 1
            and trajectory_display_count <= canvas.MAX_TRAJECTORY_DISPLAY_POINTS
            and canvas.trajectory_visible is True
            and canvas.has_sparse_map
            and sparse_display_count >= 20
            and sparse_display_count <= canvas.MAX_SPARSE_LANDMARK_DISPLAY_POINTS
            and sparse_total_count >= sparse_display_count
            and canvas.sparse_landmarks_visible is True
            and canvas.grid_visible
            and canvas.axes_visible
            and np.isfinite(world_axis_length)
            and 0.08 <= world_axis_length <= 0.34
            and orientation_gizmo_ready
            and world_axis_arrowheads
            and position is not None
            and np.isfinite(position).all()
            and retained_after_stop
            and page._viewport_placeholder.isHidden()
            and window._mode_tabs._buttons[AppMode.RECONSTRUCTION].isEnabled()
        )
        summary = {
            "phase": "8.8C3",
            "pass": passed,
            "session_id": session_id,
            "observations": len(observations),
            "canvas_has_pose": canvas.has_pose,
            "trajectory_samples_retained": canvas.trajectory_count,
            "trajectory_display_count": trajectory_display_count,
            "trajectory_visible": canvas.trajectory_visible,
            "trajectory_rendering_enabled": True,
            "sparse_map_visible": canvas.sparse_landmarks_visible,
            "sparse_map_has_geometry": canvas.has_sparse_map,
            "sparse_landmark_display_count": sparse_display_count,
            "sparse_landmark_total_count": sparse_total_count,
            "sparse_map_revision": sparse_map_revision,
            "world_axis_arrowheads": world_axis_arrowheads,
            "world_axis_length": world_axis_length,
            "orientation_gizmo_ready": orientation_gizmo_ready,
            "orientation_gizmo_projection": orientation_projection.tolist(),
            "grid_visible": canvas.grid_visible,
            "axes_visible": canvas.axes_visible,
            "current_camera_position_viewer": None if position is None else [float(v) for v in position],
            "viewport_state": page._viewport_state.text(),
            "retained_after_stop": retained_after_stop,
            "terminal_ui_ready": terminal_ui_ready,
            "cleanup_ok": cleanup_ok,
            "page_live_state_after_cleanup": page._live_state.value,
            "controller_live_state_after_cleanup": controller.live_state.value,
            "viewport_placeholder_hidden": page._viewport_placeholder.isHidden(),
            "reconstruction_navigation_enabled": window._mode_tabs._buttons[AppMode.RECONSTRUCTION].isEnabled(),
            "viewer_frame": "X=reconstruction X, Y=reconstruction Z, Z=-reconstruction Y",
            "pose_convention": "T_reconstruction_from_camera",
            "scale_status": "arbitrary",
            "certified_metrology": False,
            "backend_autostart_from_navigation": False,
            "screenshot": str(screenshot_path),
            "errors": errors,
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Phase 8.8C3 live sparse-map smoke: {'PASS' if passed else 'FAIL'}")
        print(
            f"session_id={session_id} observations={len(observations)} "
            f"visual_samples={canvas.trajectory_count} sparse_points={sparse_display_count} viewport={page._viewport_state.text()!r}"
        )
        print(f"screenshot={screenshot_path}")
        print(f"summary={summary_path}")
        return 0 if passed else 1
    finally:
        window.close()
        app.processEvents()


if __name__ == "__main__":
    raise SystemExit(main())
