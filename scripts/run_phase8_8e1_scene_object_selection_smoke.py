#!/usr/bin/env python3
"""Automated Qt smoke for Phase 8.8E1 target binding; no backend execution."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.reconstruction_inspection_selection import ReconstructionInspectionSelectionScope
from app.reconstruction_persistence_controller import ReconstructionSceneAvailability
from app.widgets.mode_tabs import AppMode

OUTPUT = Path("/tmp/phase8_8e1_scene_object_selection")


def _object(index: int, class_name: str, role: str, x: float) -> dict[str, object]:
    return {
        "object_id": f"e1-smoke-{index:04d}",
        "class_name": class_name,
        "semantic_role": role,
        "confidence": 0.92,
        "vertex_count": 240 + index * 20,
        "face_count": 150 + index * 10,
        "observation_ids": [f"e1-smoke-obs-{index}-{j}" for j in range(3)],
        "mean_vertex_quality": 0.87,
        "median_vertex_uncertainty_reconstruction_units": 0.05,
        "maximum_vertex_uncertainty_reconstruction_units": 0.09,
        "centroid_reconstruction": [x, 0.0, 2.0],
        "bounds_min_reconstruction": [x - 0.18, -0.20, 1.82],
        "bounds_max_reconstruction": [x + 0.18, 0.20, 2.18],
        "orientation_reliable": True,
        "metric_geometry_status": "arbitrary",
        "certified_metrology": False,
    }


def _payload() -> dict[str, object]:
    return {
        "session_id": "99999999-9999-4999-8999-999999999991",
        "metric_geometry_status": "arbitrary",
        "certified_metrology": False,
        "statistics": {
            "observations_total": 6,
            "observations_accepted": 6,
            "persistent_object_count": 2,
        },
        "objects": [
            _object(0, "gear", "rotating_component", -0.25),
            _object(1, "shaft", "supporting_component", 0.35),
        ],
    }


def main() -> int:
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    page.set_scene_availability(ReconstructionSceneAvailability.PARTIAL)
    page.set_semantic_geometry(_payload(), source="e1-generic-smoke")

    scene_selection = page.inspection_selection
    scene_scope_ok = scene_selection.scope is ReconstructionInspectionSelectionScope.SCENE
    page._semantic_object_combo.setCurrentIndex(2)
    object_selection = page.inspection_selection
    object_scope_ok = (
        object_selection.scope is ReconstructionInspectionSelectionScope.OBJECT
        and object_selection.object_id == "e1-smoke-0001"
        and object_selection.class_name == "shaft"
        and object_selection.semantic_role == "supporting_component"
    )
    page._semantic_clear_button.click()
    clear_returns_scene = (
        page.inspection_selection.scope is ReconstructionInspectionSelectionScope.SCENE
    )
    page._semantic_object_combo.setCurrentIndex(1)

    window.resize(1366, 768)
    window.show()
    app.processEvents()
    screenshot = OUTPUT / "reconstruction_e1_scene_object_selection.png"
    screenshot_ok = bool(window.grab().save(str(screenshot)))

    final_selection = page.inspection_selection
    summary = {
        "phase": "8.8E1",
        "pass": bool(
            scene_scope_ok
            and object_scope_ok
            and clear_returns_scene
            and screenshot_ok
            and final_selection.has_object
            and not final_selection.certified_metrology
        ),
        "scene_scope_before_object": scene_scope_ok,
        "object_scope_after_d5_selection": object_scope_ok,
        "clear_returns_scene_scope": clear_returns_scene,
        "semantic_selector_reused": True,
        "duplicate_inspection_object_selector": False,
        "selected_object_id": final_selection.object_id,
        "selected_class": final_selection.class_name,
        "selected_role": final_selection.semantic_role,
        "canonical_geometry_authoritative": final_selection.canonical_geometry_authoritative,
        "metric_geometry_status": final_selection.metric_geometry_status,
        "certified_metrology": final_selection.certified_metrology,
        "measurement_evaluated": False,
        "qc_evaluated": False,
        "reference_comparison_evaluated": False,
        "authority_ui": page._inspection_authority_value.text(),
        "state_ui": page._inspection_state_value.text(),
        "screenshot": str(screenshot),
        "errors": [],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    window.close()
    app.processEvents()

    if summary["pass"]:
        print("Phase 8.8E1 scene/object selection smoke: PASS")
    else:
        print("Phase 8.8E1 scene/object selection smoke: FAIL")
    print(
        f"scope={final_selection.scope.value!r} object={final_selection.object_id!r} "
        f"class={final_selection.class_name!r} role={final_selection.semantic_role!r}"
    )
    print(f"authority={summary['authority_ui']!r}")
    print(f"screenshot={screenshot}")
    print(f"summary={OUTPUT / 'summary.json'}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
