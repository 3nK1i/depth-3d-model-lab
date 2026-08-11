#!/usr/bin/env python3
"""Qualified Phase 8.7B semantics -> Phase 8.8D5 Qt/VisPy presentation smoke."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode


def _obj_vertices(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("v "):
            continue
        parts = line.split()
        if len(parts) >= 4:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(vertices, dtype=np.float64).reshape((-1, 3))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/phase8_8d5_semantic_presentation"),
    )
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    # Re-run the already-qualified Phase 8.7B semantic geometry smoke so D5's
    # decisive gate consumes real semantic-engine output, not a UI-only mock.
    semantic_source = Path("/tmp/phase8_7b_semantic_geometry")
    shutil.rmtree(semantic_source, ignore_errors=True)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_semantic_geometry_smoke.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Phase 8.7B semantic geometry smoke failed before D5 presentation:\n"
            + completed.stdout
            + completed.stderr
        )
    source_summary_path = semantic_source / "summary.json"
    source_obj_path = semantic_source / "semantic_groups.obj"
    if not source_summary_path.is_file() or not source_obj_path.is_file():
        raise RuntimeError("Qualified semantic smoke did not produce its expected artifacts.")
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    semantic_payload = source_summary.get("semantic", {})
    if source_summary.get("pass") is not True or not isinstance(semantic_payload, dict):
        raise RuntimeError("Qualified semantic smoke summary is not a valid PASS payload.")

    os.environ.pop("DEPTH3D_TEST_FAKE_VISPY", None)
    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = MainWindow()
    window.show()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    app.processEvents()
    page = window._reconstruction_page
    canvas = page._reconstruction_canvas
    errors: list[str] = []
    window._reconstruction_controller.error_occurred.connect(errors.append)

    try:
        page.set_semantic_geometry(semantic_payload, source="phase8_7b_semantic_geometry_smoke")
        # Present the exact Phase 8.7B surface vertices as non-authoritative
        # context behind the semantic support boxes; no new reconstruction is run.
        vertices = _obj_vertices(source_obj_path)
        canvas.update_sparse_landmarks(vertices, total_count=len(vertices))
        if page._semantic_object_combo.count() > 2:
            page._semantic_object_combo.setCurrentIndex(2)
        canvas.fit_scene()
        app.processEvents()

        selected_index = canvas.selected_semantic_object_index
        selected = None
        presentation = page._offline_semantics
        if presentation is not None and selected_index is not None:
            selected = presentation.objects[selected_index]

        semantic_button = page._viewport_tool_buttons["semantic_objects"]
        initial_visible = canvas.semantic_objects_visible
        semantic_button.setChecked(False)
        app.processEvents()
        hidden_ok = not canvas.semantic_objects_visible
        semantic_button.setChecked(True)
        app.processEvents()
        restored_ok = canvas.semantic_objects_visible and canvas.selected_semantic_object_visible

        screenshot = output / "reconstruction_semantic_objects.png"
        window.grab().save(str(screenshot))

        authority_text = page._semantic_authority_value.text()
        scale_text = page._semantic_scale_value.text()
        quality_text = page._semantic_quality_value.text()
        support_text = page._semantic_support_value.text()
        role_text = page._semantic_role_value.text()
        semantic_count = 0 if presentation is None else presentation.total_object_count
        displayed = canvas.semantic_object_display_count

        passed = bool(
            source_summary.get("pass") is True
            and presentation is not None
            and presentation.detailed_snapshot
            and presentation.metric_geometry_status == "arbitrary"
            and presentation.certified_metrology is False
            and semantic_count == 3
            and displayed == 3
            and canvas.semantic_object_ids
            and selected is not None
            and canvas.selected_semantic_object_id == selected.object_id
            and canvas.selected_semantic_object_visible
            and page._semantic_objects_group.isHidden() is False
            and initial_visible
            and hidden_ok
            and restored_ok
            and "Semantic context" in authority_text
            and "inspection/QC" in authority_text
            and "Arbitrary reconstruction gauge" in scale_text
            and "Semantic " in quality_text
            and "obs" in support_text
            and role_text != "—"
            and "measurement accuracy" in page._semantic_quality_value.toolTip()
            and semantic_button.isEnabled()
            and not source_summary.get("semantic", {}).get("certified_metrology", True)
            and vertices.shape[0] > 0
            and not errors
        )
        summary = {
            "phase": "8.8D5",
            "pass": passed,
            "source_phase": source_summary.get("phase"),
            "source_semantic_smoke_pass": source_summary.get("pass"),
            "semantic_session_id": None if presentation is None else presentation.session_id,
            "semantic_object_total_count": semantic_count,
            "semantic_object_display_count": displayed,
            "semantic_objects_visible": canvas.semantic_objects_visible,
            "selected_semantic_object_index": canvas.selected_semantic_object_index,
            "selected_semantic_object_id": canvas.selected_semantic_object_id,
            "selected_semantic_class": None if selected is None else selected.class_name,
            "selected_semantic_role": None if selected is None else selected.semantic_role,
            "selected_semantic_visible": canvas.selected_semantic_object_visible,
            "semantic_role_ui": role_text,
            "semantic_support_ui": support_text,
            "semantic_quality_ui": quality_text,
            "semantic_scale_ui": scale_text,
            "semantic_authority_ui": authority_text,
            "metric_geometry_status": None if presentation is None else presentation.metric_geometry_status,
            "certified_metrology": False if presentation is None else presentation.certified_metrology,
            "measurement_claimed": False,
            "inspection_decision_claimed": False,
            "semantic_toggle_hide_restore": hidden_ok and restored_ok,
            "surface_vertex_count": int(vertices.shape[0]),
            "sparse_context_display_count": canvas.sparse_landmark_display_count,
            "viewport_state": page._viewport_state.text(),
            "screenshot": str(screenshot),
            "errors": errors,
        }
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Phase 8.8D5 semantic/object presentation smoke: {'PASS' if passed else 'FAIL'}")
        print(
            f"objects={semantic_count} displayed={displayed} selected={canvas.selected_semantic_object_id!r} "
            f"role={role_text!r}"
        )
        print(f"authority={authority_text!r} scale={scale_text!r}")
        print(f"screenshot={screenshot}")
        print(f"summary={summary_path}")
        return 0 if passed else 1
    finally:
        window.close()
        app.processEvents()


if __name__ == "__main__":
    raise SystemExit(main())
