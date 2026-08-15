#!/usr/bin/env python3
"""Real Qt/VisPy preview of qualified Phase 9.3J measurement registration.

Uses deterministic persisted canonical/measurement evidence only. No depth,
perception, reconstruction, DA3, Gaussian, or NeRF backend is launched.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

os.environ.pop("DEPTH3D_TEST_FAKE_VISPY", None)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from core.reconstruction.measurement_registration import CanonicalMeasurementRegistrationState
from tests.phase9_3j_helpers import (
    OBSERVATION_ID,
    make_phase9_3j_inspection_bundle,
    make_phase9_3j_qualified_scene,
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/phase9_3j_measurement_registration_preview.png"),
    )
    p.add_argument("--show", action="store_true")
    return p


def _pump(app: QApplication, count: int = 16) -> None:
    for _ in range(count):
        app.processEvents()
        time.sleep(0.02)


def main() -> int:
    args = _parser().parse_args()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    with tempfile.TemporaryDirectory(prefix="phase9_3j_measurement_preview_") as tmp:
        root = Path(tmp)
        scene, authority = make_phase9_3j_qualified_scene(root / "scene")
        summary, _obj = make_phase9_3j_inspection_bundle(root / "inspection")

        window = MainWindow()
        window.resize(1366, 768)
        window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
        page = window._reconstruction_page
        inspection = window._inspection_page
        page.set_persistent_scene(scene, "phase9_3j_qualified_scene.json")
        page.set_persistent_authority_snapshot(authority, "phase9_3j_authority.json")

        height, width = summary.image_height, summary.image_width
        arrays = {
            "aligned_rgb": np.zeros((height, width, 3), dtype=np.uint8),
            "mask:0": np.zeros((height, width), dtype=bool),
            "raw:0": np.zeros((0, 3), dtype=np.float32),
            "filtered:0": np.zeros((0, 3), dtype=np.float32),
        }
        inspection.load_bundle(summary, arrays)
        semantic = page.canonical_semantic_presentation
        if semantic is None:
            raise RuntimeError("Qualified preview did not load canonical semantics.")
        target = next((item for item in semantic.objects if OBSERVATION_ID in item.observation_ids), None)
        if target is None or not page.select_semantic_object_by_id(target.object_id):
            raise RuntimeError("Qualified preview could not bind the exact measurement observation to canonical semantics.")
        window._refresh_measurement_canonical_registration()

        presentation = inspection.canonical_registration_presentation
        if presentation is None or presentation.state is not CanonicalMeasurementRegistrationState.QUALIFIED:
            raise RuntimeError("Qualified preview did not produce qualified canonical registration.")
        if len(presentation.dimensions) != 1:
            raise RuntimeError("Qualified preview did not produce exactly one source-authored dimension.")
        canvas = page._reconstruction_canvas
        if canvas.canonical_registered_dimension_count != 1 or not canvas.canonical_registered_dimensions_visible:
            raise RuntimeError("Qualified preview did not render the registered canonical dimension segment.")
        if presentation.dimensions[0].registered_dimension_uncertainty_si < presentation.dimensions[0].source_expanded_uncertainty_si:
            raise RuntimeError("Canonical registration incorrectly reduced source measurement uncertainty.")
        if presentation.certified_metrology or scene.certified_metrology:
            raise RuntimeError("Phase 9.3J preview incorrectly claimed certified metrology.")

        page._canonical_registration_group.set_expanded(True)
        window.show()
        page._settings_scroll.ensureWidgetVisible(page._canonical_registration_group)
        _pump(app, 20)

        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if not window.grab().save(str(output), "PNG"):
            raise RuntimeError(f"Unable to save 9.3J measurement preview screenshot: {output}")

        width_dim = presentation.dimensions[0]
        print(f"Phase 9.3J qualified measurement registration preview: {output}")
        print(
            f"state={presentation.state.value} object={presentation.canonical_object_id} "
            f"camera={presentation.canonical_camera_id} dimension_count={len(presentation.dimensions)}"
        )
        print(
            f"source_width_mm={width_dim.value_si * 1000.0:.1f} "
            f"source_uncertainty_mm={width_dim.source_expanded_uncertainty_si * 1000.0:.2f} "
            f"registered_uncertainty_mm={width_dim.registered_dimension_uncertainty_si * 1000.0:.2f} "
            f"placement_uncertainty_mm={width_dim.endpoint_placement_uncertainty_si * 1000.0:.2f}"
        )
        print("Original Phase 7.5 measurement remains source truth; canonical placement adds uncertainty and does not certify metrology.")
        print("Calibration is optional: this qualified fixture uses explicit validated metric-scale evidence without requiring a camera-calibration profile.")
        print("No depth/perception/reconstruction/DA3/Gaussian/NeRF backend was launched by this preview.")

        if args.show:
            code = app.exec()
            window.close()
            print("Phase 9.3J measurement-registration viewer resources released on shutdown.")
            return int(code)
        window.close()
        print("Phase 9.3J measurement-registration viewer resources released on shutdown.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
