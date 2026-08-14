#!/usr/bin/env python3
"""Real-VisPy preview for Phase 9.3E object inspection + measurement overlays."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
import time

# Exercise the qualified real VisPy/OpenGL canvas in the production environment.
os.environ.pop("DEPTH3D_TEST_FAKE_VISPY", None)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.widgets.reconstruction_page import ReconstructionPage
from tests.phase9_3d_helpers import make_phase9_3d_scene
from tests.phase9_3e_helpers import inspection_report, measurement_context, measurement_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/phase9_3e_canonical_inspection_preview.png"),
        help="PNG screenshot path",
    )
    parser.add_argument("--show", action="store_true", help="Keep the real VisPy preview open.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    with tempfile.TemporaryDirectory(prefix="phase9_3e_preview_") as tmp:
        root = Path(tmp).resolve()
        scene, semantic, _surface = make_phase9_3d_scene(
            root / "source", request_id="phase9-3e-preview"
        )
        page = ReconstructionPage()
        page.resize(1366, 768)
        page.set_persistent_scene(scene, "phase9_3e_preview/persistent_reconstruction_scene.json")
        page._canonical_scene_group.set_expanded(True)
        page._semantic_objects_group.set_expanded(True)
        page.show()
        for _ in range(15):
            app.processEvents()
            time.sleep(0.02)

        canvas = page._reconstruction_canvas
        if page.canonical_semantic_presentation is None or not canvas.canonical_semantic_loaded:
            raise RuntimeError("Phase 9.3E preview did not bind the persisted 9.3D semantic scene.")

        page._semantic_object_combo.setCurrentIndex(1)
        first_object_id = semantic.objects[0].object_id
        evidence = measurement_payload()

        # E2 alone must not expose unqualified numeric values in the 3D viewport.
        page.set_reconstruction_measurement_evidence(evidence)
        if canvas.canonical_inspection_overlay_visible:
            raise RuntimeError("9.3E exposed a numeric viewport overlay before E3 validity evidence.")

        page.set_reconstruction_measurement_validity_evidence(
            evidence, context=measurement_context()
        )
        page.set_reconstruction_inspection_qc_evidence(
            inspection_report(
                session_id=scene.session_id,
                object_id=first_object_id,
                include_object=True,
            )
        )
        for _ in range(10):
            app.processEvents()
            time.sleep(0.02)

        overlay = page.canonical_inspection_overlay_presentation
        if overlay is None or not canvas.canonical_inspection_overlay_visible:
            raise RuntimeError("Phase 9.3E qualified source-evidence overlay is not visible.")
        if "±" not in overlay.overlay_text or "REJECTED" not in overlay.overlay_text:
            raise RuntimeError("Phase 9.3E overlay lost uncertainty/rejection semantics.")
        if "QC: WARN" not in overlay.overlay_text:
            raise RuntimeError("Phase 9.3E overlay lost exact selected-object QC evidence.")

        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if not page.grab().save(str(output), "PNG"):
            raise RuntimeError(f"Unable to save preview screenshot: {output}")

        print(f"Phase 9.3E real-VisPy inspection overlay preview: {output}")
        print(
            f"selected_object={overlay.object_id} numeric_available={overlay.numeric_measurement_count} "
            f"rejected={overlay.rejected_measurement_count} qc_exact={overlay.inspection_result_available}"
        )
        print(
            "Numeric viewport values require E3 validity/uncertainty; rejected/unavailable values expose no numbers."
        )
        print(
            "Semantic authority=semantic_context; canonical geometry authority unchanged; "
            "measurement/QC remain source evidence; certified_metrology=False."
        )
        print(
            "No measurement box/arrows were fabricated from semantic bounds or camera-space Phase 7.5 geometry."
        )
        print("No YOLO/SAM2/reconstruction/DA3/Gaussian/NeRF backend was launched by scene loading.")

        if args.show:
            exit_code = app.exec()
            page.shutdown_viewport()
            if canvas.canonical_inspection_overlay_visible or canvas.canonical_semantic_loaded or canvas.canonical_mesh_loaded:
                raise RuntimeError("Canonical inspection/semantic resources remained loaded after shutdown.")
            page.close()
            print("Canonical inspection/semantic/canonical-mesh resources released on viewport shutdown.")
            return int(exit_code)

        page.shutdown_viewport()
        if canvas.canonical_inspection_overlay_visible or canvas.canonical_semantic_loaded or canvas.canonical_mesh_loaded:
            raise RuntimeError("Canonical inspection/semantic resources remained loaded after shutdown.")
        page.close()
        print("Canonical inspection/semantic/canonical-mesh resources released on viewport shutdown.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
