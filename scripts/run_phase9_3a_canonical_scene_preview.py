#!/usr/bin/env python3
"""UI-only Phase 9.3A preview; no reconstruction or neural backend is launched."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("DEPTH3D_TEST_FAKE_VISPY", "1")
os.environ.setdefault("QT_OPENGL", "software")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.widgets.reconstruction_page import ReconstructionPage
from core.reconstruction import (
    MetricEvidenceValidationState,
    build_persistent_scene_authority_snapshot,
    persistent_scene_from_pipeline_result,
    reference_target_metric_evidence,
)
from core.reconstruction.calibration_management import create_manual_camera_calibration_profile
from tests.reconstruction_f5_helpers import run_f5_fixture


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/phase9_3a_canonical_scene_preview.png"),
        help="PNG screenshot path (default: /tmp/phase9_3a_canonical_scene_preview.png)",
    )
    parser.add_argument("--show", action="store_true", help="Keep the preview window open interactively.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    with tempfile.TemporaryDirectory(prefix="phase9_3a_preview_") as tmp:
        root = Path(tmp).resolve()
        _fixture, _providers, result, snapshots = run_f5_fixture(
            root / "run", request_id="phase9-3a-preview"
        )
        surface = (root / "canonical_surface.obj").resolve()
        surface.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
        snapshots = dict(snapshots)
        snapshots["surface_mesh"] = {
            "surface_obj_path": str(surface),
            "metric_geometry_status": "arbitrary",
            "canonical_geometry_authoritative": True,
            "certified_metrology": False,
        }
        scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
        profile = create_manual_camera_calibration_profile(
            width=640,
            height=480,
            fx=520.0,
            fy=521.0,
            cx=320.0,
            cy=240.0,
            profile_id="phase9-3a-preview-profile",
            calibration_record_id="phase9-3a-preview-record",
            camera_id="phase9-3a-preview-camera",
            provenance="phase9_3a_preview",
        )
        evidence = reference_target_metric_evidence(
            evidence_id="phase9-3a-preview-target",
            reconstruction_frame_id="reconstruction-world",
            reference_id="phase9-3a-preview-target-ref",
            target_length_m=0.25,
            measured_length_reconstruction_units=5.0,
            target_relative_uncertainty=0.005,
            measurement_relative_uncertainty=0.02,
            provenance="phase9_3a_preview",
            validation_state=MetricEvidenceValidationState.VALIDATED,
            validation_record_id="phase9-3a-preview-validation",
        )
        authority = build_persistent_scene_authority_snapshot(
            scene,
            camera_calibration_profile=profile,
            metric_qualification_evidence=evidence,
        )

        page = ReconstructionPage()
        page.resize(1366, 768)
        page.set_persistent_scene(scene, "phase9_3a_preview/persistent_reconstruction_scene.json")
        page.set_persistent_authority_snapshot(authority, "phase9_3a_preview/persistent_scene_authority.json")
        page._canonical_scene_group.set_expanded(True)
        page.show()
        app.processEvents()

        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if not page.grab().save(str(output), "PNG"):
            raise RuntimeError(f"Unable to save preview screenshot: {output}")
        print(f"Phase 9.3A canonical-scene preview: {output}")
        print("Synthetic UI preview only; no reconstruction/DA3/Gaussian/NeRF backend was launched.")

        if args.show:
            return app.exec()
        page.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
