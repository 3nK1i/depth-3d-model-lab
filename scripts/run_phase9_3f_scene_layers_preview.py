#!/usr/bin/env python3
"""Real-VisPy preview for Phase 9.3F professional scene layers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.pop("DEPTH3D_TEST_FAKE_VISPY", None)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.widgets.reconstruction_page import ReconstructionPage
from tests.phase9_3e_helpers import inspection_report, measurement_context, measurement_payload
from tests.phase9_3f_helpers import make_phase9_3f_scene


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/phase9_3f_scene_layers_preview.png"),
        help="PNG screenshot path",
    )
    parser.add_argument("--show", action="store_true", help="Keep the real VisPy preview open.")
    return parser


def _pump(app: QApplication, count: int = 10) -> None:
    for _ in range(count):
        app.processEvents()
        time.sleep(0.02)


def main() -> int:
    args = _parser().parse_args()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    with tempfile.TemporaryDirectory(prefix="phase9_3f_preview_") as tmp:
        root = Path(tmp).resolve()
        scene, semantic, _surface = make_phase9_3f_scene(
            root / "source", request_id="phase9-3f-preview"
        )
        page = ReconstructionPage()
        page.resize(1366, 768)
        page.set_persistent_scene(scene, "phase9_3f_preview/persistent_reconstruction_scene.json")
        page._semantic_object_combo.setCurrentIndex(1)
        object_id = semantic.objects[0].object_id
        evidence = measurement_payload()
        page.set_reconstruction_measurement_evidence(evidence)
        page.set_reconstruction_measurement_validity_evidence(evidence, context=measurement_context())
        page.set_reconstruction_inspection_qc_evidence(
            inspection_report(session_id=scene.session_id, object_id=object_id, include_object=True)
        )
        page._scene_layers_group.set_expanded(True)
        page.show()
        page._settings_scroll.ensureWidgetVisible(page._scene_layers_group)
        _pump(app, 15)

        canvas = page._reconstruction_canvas
        layers = page._scene_layer_toggles
        expected = {
            "canonical_surface", "grid", "axes", "camera_frustums", "camera_trajectory",
            "sparse_landmarks", "sparse_envelope", "semantic_objects", "semantic_labels",
            "semantic_selection", "inspection_evidence",
        }
        if set(layers) != expected:
            raise RuntimeError("Phase 9.3F preview did not build the complete scene-layer stack.")
        if not (canvas.canonical_mesh_loaded and canvas.canonical_camera_loaded and canvas.canonical_semantic_loaded):
            raise RuntimeError("Phase 9.3F preview is missing canonical scene resources.")
        if page.canonical_inspection_overlay_presentation is None:
            raise RuntimeError("Phase 9.3F preview is missing qualified 9.3E inspection evidence.")

        # Exercise every new independent canonical layer off/on once. Resources
        # must remain loaded because 9.3F controls presentation, not evidence.
        for key in ("canonical_surface", "axes", "semantic_labels", "semantic_selection", "inspection_evidence"):
            layers[key].setChecked(False)
            _pump(app, 2)
            layers[key].setChecked(True)
            _pump(app, 2)
        if not (canvas.canonical_mesh_loaded and canvas.canonical_camera_loaded and canvas.canonical_semantic_loaded):
            raise RuntimeError("Phase 9.3F visibility round-trip unloaded scene evidence.")
        if scene.certified_metrology or not scene.canonical_geometry_authoritative:
            raise RuntimeError("Phase 9.3F visibility round-trip changed scene authority.")

        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if not page.grab().save(str(output), "PNG"):
            raise RuntimeError(f"Unable to save preview screenshot: {output}")

        print(f"Phase 9.3F real-VisPy scene-layer preview: {output}")
        print(
            f"layers={len(layers)} canonical_surface={canvas.canonical_mesh_visible} "
            f"cameras={canvas.canonical_camera_count} semantics={canvas.semantic_object_display_count} "
            f"inspection_overlay={canvas.canonical_inspection_overlay_visible}"
        )
        print("Scene-layer visibility is presentation-only; loaded evidence and authority remain unchanged.")
        print("Existing grid/camera/trajectory/landmark/envelope/semantic quick toggles remain mirrored.")
        print("No YOLO/SAM2/reconstruction/DA3/Gaussian/NeRF backend was launched by scene loading.")

        if args.show:
            exit_code = app.exec()
            page.shutdown_viewport()
            if canvas.canonical_mesh_loaded or canvas.canonical_camera_loaded or canvas.canonical_semantic_loaded:
                raise RuntimeError("Canonical scene resources remained loaded after viewport shutdown.")
            page.close()
            print("Canonical scene-layer resources released on viewport shutdown.")
            return int(exit_code)

        page.shutdown_viewport()
        if canvas.canonical_mesh_loaded or canvas.canonical_camera_loaded or canvas.canonical_semantic_loaded:
            raise RuntimeError("Canonical scene resources remained loaded after viewport shutdown.")
        page.close()
        print("Canonical scene-layer resources released on viewport shutdown.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
