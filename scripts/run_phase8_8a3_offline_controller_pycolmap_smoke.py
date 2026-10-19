#!/usr/bin/env python3
"""Real non-UI Phase 8.8A3 smoke through ReconstructionController + PyCOLMAP."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from time import monotonic, sleep

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_OPENGL", "software")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.application_controller import ApplicationController
from app.reconstruction_controller import (
    OfflineReconstructionConfiguration,
    OfflineReconstructionRunState,
    OfflineReconstructionSourceKind,
    ReconstructionController,
)
from app.widgets.original_view import OriginalSourceView
from app.widgets.source_bar import SourceBar
from core.reconstruction import ReconstructionQualityProfile, SfmMatchingStrategy


def _default_sfm_python() -> Path:
    return Path.home() / "anaconda3/envs/depth3d-pycolmap-probe/bin/python"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path, help="Directory of overlapping images.")
    source.add_argument("--video", type=Path, help="Recorded video with camera motion/overlap.")
    parser.add_argument("--sfm-python", type=Path, default=_default_sfm_python())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/phase8_8a3_offline_controller_pycolmap"),
    )
    parser.add_argument(
        "--profile",
        choices=[item.value for item in ReconstructionQualityProfile],
        default=ReconstructionQualityProfile.EFFICIENT.value,
    )
    parser.add_argument(
        "--matching",
        choices=[item.value for item in SfmMatchingStrategy],
        default=SfmMatchingStrategy.SEQUENTIAL.value,
    )
    parser.add_argument("--max-keyframes", type=int, default=50)
    parser.add_argument("--maximum-image-size", type=int, default=None)
    parser.add_argument("--sequential-overlap", type=int, default=None)
    parser.add_argument("--video-target-fps", type=float, default=2.0)
    parser.add_argument("--image-sequence-fps", type=float, default=10.0)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--backend-timeout-minutes", type=float, default=60.0)
    parser.add_argument("--controller-timeout-minutes", type=float, default=70.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        import shutil

        shutil.rmtree(output)

    if args.images is not None:
        kind = OfflineReconstructionSourceKind.IMAGE_DIRECTORY
        inputs = (args.images.expanduser().resolve(),)
    else:
        kind = OfflineReconstructionSourceKind.RECORDED_VIDEO
        inputs = (args.video.expanduser().resolve(),)

    app = QApplication.instance() or QApplication([])
    source_bar = SourceBar()
    original_view = OriginalSourceView()
    application_controller = ApplicationController(source_bar, original_view)
    controller = ReconstructionController(application_controller, parent=application_controller)
    states: list[str] = []
    events: list[dict] = []
    errors: list[str] = []
    results: list[dict] = []
    controller.offline_state_changed.connect(lambda state: states.append(state.value))
    controller.offline_event_received.connect(events.append)
    controller.error_occurred.connect(errors.append)
    controller.offline_result_ready.connect(lambda payload, _out: results.append(payload))

    configuration = OfflineReconstructionConfiguration(
        source_kind=kind,
        inputs=inputs,
        output_dir=output,
        sfm_python=args.sfm_python,
        profile=ReconstructionQualityProfile(args.profile),
        matching=SfmMatchingStrategy(args.matching),
        max_keyframes=args.max_keyframes,
        maximum_image_size=args.maximum_image_size,
        sequential_overlap=args.sequential_overlap,
        video_target_fps=args.video_target_fps,
        image_sequence_fps=args.image_sequence_fps,
        recursive=args.recursive,
        use_gpu=not args.cpu,
        timeout_minutes=args.backend_timeout_minutes,
    )

    try:
        session_id = controller.start_offline(configuration)
        deadline = monotonic() + float(args.controller_timeout_minutes) * 60.0
        while controller.offline_state not in {
            OfflineReconstructionRunState.COMPLETED,
            OfflineReconstructionRunState.FAILED,
            OfflineReconstructionRunState.CANCELLED,
        }:
            app.processEvents()
            if monotonic() >= deadline:
                controller.cancel_offline()
                raise RuntimeError("A3 controller smoke exceeded its bounded controller timeout.")
            sleep(0.01)
        app.processEvents()

        result_path = output / "offline_result.json"
        if not result_path.is_file():
            raise RuntimeError("Offline worker did not produce offline_result.json.")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        passed = bool(
            controller.offline_state is OfflineReconstructionRunState.COMPLETED
            and payload.get("pass") is True
            and payload.get("session_id") == session_id
            and payload.get("production_input_assembly") is True
            and payload.get("tests_fixture_used") is False
            and payload.get("camera_pose_available") is True
            and payload.get("canonical_geometry_authoritative") is False
            and payload.get("persistent_scene_ready") is False
            and payload.get("certified_metrology") is False
            and payload.get("reconstruction_ui_enabled") is False
            and not errors
        )
        summary = {
            "phase": "8.8A3",
            "pass": passed,
            "controller": "ReconstructionController",
            "controller_state": controller.offline_state.value,
            "session_id": session_id,
            "source_kind": payload.get("source_kind"),
            "states": states,
            "event_count": len(events),
            "errors": errors,
            "production_input_assembly": payload.get("production_input_assembly"),
            "tests_fixture_used": payload.get("tests_fixture_used"),
            "pycolmap_external_process": payload.get("execution_boundary", {}).get("pycolmap_external_process"),
            "production_depth_imports_pycolmap": payload.get("execution_boundary", {}).get("production_depth_imports_pycolmap"),
            "registered_keyframes": payload.get("sfm_summary", {}).get("registered_keyframes"),
            "landmarks": payload.get("sfm_summary", {}).get("landmarks"),
            "gauge": payload.get("sfm_summary", {}).get("gauge"),
            "canonical_geometry_authoritative": payload.get("canonical_geometry_authoritative"),
            "persistent_scene_ready": payload.get("persistent_scene_ready"),
            "reconstruction_ui_enabled": payload.get("reconstruction_ui_enabled"),
            "offline_result": str(result_path.resolve()),
        }
        (output / "controller_smoke_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"Phase 8.8A3 ReconstructionController offline PyCOLMAP smoke: {'PASS' if passed else 'FAIL'}"
        )
        print(f"session_id={session_id} state={controller.offline_state.value}")
        print(
            f"source_kind={summary['source_kind']} registered={summary['registered_keyframes']} "
            f"landmarks={summary['landmarks']}"
        )
        print(
            "production_input_assembly="
            f"{summary['production_input_assembly']} tests_fixture_used={summary['tests_fixture_used']}"
        )
        print(
            "pycolmap_external_process="
            f"{summary['pycolmap_external_process']} production_depth_imports_pycolmap="
            f"{summary['production_depth_imports_pycolmap']}"
        )
        print(
            "canonical_geometry_authoritative="
            f"{summary['canonical_geometry_authoritative']} persistent_scene_ready="
            f"{summary['persistent_scene_ready']} reconstruction_ui_enabled={summary['reconstruction_ui_enabled']}"
        )
        print(f"summary={output / 'controller_smoke_summary.json'}")
        return 0 if passed else 1
    finally:
        controller.shutdown()
        application_controller.shutdown()
        app.processEvents()


if __name__ == "__main__":
    raise SystemExit(main())
