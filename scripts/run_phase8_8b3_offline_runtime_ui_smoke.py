#!/usr/bin/env python3
"""Real B3 UI/controller offline-runtime smoke using an overlapping image sequence."""

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
from app.reconstruction_controller import ReconstructionWorkflow
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
        default=Path("/tmp/phase8_8b3_offline_runtime_ui"),
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
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_OPENGL", "software")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("DEPTH3D_TEST_FAKE_VISPY", "1")

    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = MainWindow()
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    try:
        selection = controller.select_image_sequence(images)
        app.processEvents()
        if selection.workflow is not ReconstructionWorkflow.IMAGE_SEQUENCE or not selection.ready:
            raise RuntimeError("B3 image-sequence source did not reach Ready state.")
        if not page._start_button.isEnabled():
            raise RuntimeError("B3 Start button did not enable for a ready image sequence.")
        if window._mode_tabs._buttons[AppMode.RECONSTRUCTION].isEnabled():
            raise RuntimeError("Reconstruction navigation became enabled during B3.")

        page._start_button.click()
        app.processEvents()
        session_id = controller.offline_session_id
        if not session_id:
            raise RuntimeError("B3 Start did not create an offline reconstruction session.")

        deadline = monotonic() + float(args.timeout_minutes) * 60.0
        terminal = {
            OfflineReconstructionRunState.COMPLETED,
            OfflineReconstructionRunState.FAILED,
            OfflineReconstructionRunState.CANCELLED,
        }
        while controller.offline_state not in terminal and monotonic() < deadline:
            app.processEvents()
            sleep(0.05)
        app.processEvents()
        state = controller.offline_state
        if state not in terminal:
            controller.cancel_offline()
            raise RuntimeError("B3 offline runtime smoke timed out.")

        result_path = (controller.offline_output_dir or Path("/missing")) / "offline_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        passed = (
            state is OfflineReconstructionRunState.COMPLETED
            and bool(result.get("pass"))
            and result.get("session_id") == session_id
            and result.get("production_input_assembly") is True
            and result.get("tests_fixture_used") is False
            and result.get("execution_boundary", {}).get("production_depth_imports_pycolmap") is False
            and result.get("execution_boundary", {}).get("pycolmap_external_process") is True
            and page._start_button.isEnabled()
            and not page._pause_button.isEnabled()
            and not page._stop_button.isEnabled()
            and not window._mode_tabs._buttons[AppMode.RECONSTRUCTION].isEnabled()
            and not errors
        )
        summary = {
            "phase": "8.8B3",
            "pass": passed,
            "controller_state": state.value,
            "session_id": session_id,
            "source_kind": result.get("source_kind"),
            "registered_keyframes": result.get("sfm_summary", {}).get("registered_keyframes"),
            "landmarks": result.get("sfm_summary", {}).get("landmarks"),
            "production_input_assembly": result.get("production_input_assembly"),
            "tests_fixture_used": result.get("tests_fixture_used"),
            "pycolmap_external_process": result.get("execution_boundary", {}).get("pycolmap_external_process"),
            "production_depth_imports_pycolmap": result.get("execution_boundary", {}).get("production_depth_imports_pycolmap"),
            "start_reenabled_after_completion": page._start_button.isEnabled(),
            "pause_disabled_for_offline": not page._pause_button.isEnabled(),
            "stop_disabled_after_completion": not page._stop_button.isEnabled(),
            "reconstruction_navigation_enabled": window._mode_tabs._buttons[AppMode.RECONSTRUCTION].isEnabled(),
            "errors": errors,
            "offline_result": str(result_path),
        }
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(f"Phase 8.8B3 offline runtime UI smoke: {'PASS' if passed else 'FAIL'}")
        print(f"session_id={session_id} state={state.value}")
        print(
            "registered="
            f"{summary['registered_keyframes']} landmarks={summary['landmarks']} "
            f"start_reenabled={summary['start_reenabled_after_completion']}"
        )
        print(
            "production_input_assembly="
            f"{summary['production_input_assembly']} tests_fixture_used={summary['tests_fixture_used']}"
        )
        print(f"summary={summary_path}")
        return 0 if passed else 1
    finally:
        window.close()
        app.processEvents()


if __name__ == "__main__":
    raise SystemExit(main())
