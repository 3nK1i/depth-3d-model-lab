#!/usr/bin/env python3
"""Decisive Phase 8.8F5 resource/fallback UX smoke."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

import app.reconstruction_appearance_resources as resource_ui
from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from core.reconstruction import (
    GaussianSceneArtifact,
    NerfSceneArtifact,
    persistent_scene_from_pipeline_result,
    read_persistent_reconstruction_scene,
    write_persistent_reconstruction_scene,
)
from tests.reconstruction_f5_helpers import run_f5_fixture
from tests.reconstruction_helpers import make_hardware

OUTPUT = Path("/tmp/phase8_8f5_resource_fallback")


def _write_preview(path: Path, *, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (800, 450), (8, 11, 17))
    draw = ImageDraw.Draw(image)
    draw.rectangle((150, 82, 650, 368), outline=(82, 148, 218), width=5)
    draw.ellipse((280, 118, 520, 358), fill=(56, 104, 174))
    draw.text((18, 18), label, fill=(198, 208, 222))
    image.save(path)


def _activate(page, label: str, app: QApplication) -> None:
    index = page._appearance_representation_combo.findText(label)
    if index < 0:
        raise RuntimeError(f"appearance selector does not contain {label}")
    page._appearance_representation_combo.setCurrentIndex(index)
    page._appearance_representation_combo.activated.emit(index)
    app.processEvents()


def _persistent_scene(root: Path, *, degraded: bool, request_id: str):
    fixture, _providers, result, snapshots = run_f5_fixture(
        root / "source", degrade_gaussian=degraded, request_id=request_id
    )
    if not degraded:
        gaussian = fixture.store.find_unique(GaussianSceneArtifact)
        gaussian_preview = (Path(gaussian.artifact_dir) / "preview.png").resolve()
        _write_preview(gaussian_preview, label="F5 qualified Gaussian preview")
        (Path(gaussian.artifact_dir) / "qualification_metrics.json").write_text(
            json.dumps({"render_written": True, "preview_path": str(gaussian_preview)}),
            encoding="utf-8",
        )
    nerf = fixture.store.find_unique(NerfSceneArtifact)
    nerf_preview = (Path(nerf.artifact_dir) / "preview.png").resolve()
    _write_preview(nerf_preview, label="F5 qualified NeRF preview")
    (Path(nerf.artifact_dir) / "e2_runtime_evidence.json").write_text(
        json.dumps({"preview_path": str(nerf_preview), "training_completed_without_oom": True}),
        encoding="utf-8",
    )
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    manifest = write_persistent_reconstruction_scene(scene, root / "persistent")
    return read_persistent_reconstruction_scene(manifest), manifest, fixture


def main() -> int:
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Deterministic F5 proof: an existing static preview must remain displayable
    # even on a CPU-only application resource snapshot.
    resource_ui.detect_hardware = lambda: make_hardware(total_gib=None)  # type: ignore[assignment]

    full_scene, full_manifest, full_fixture = _persistent_scene(
        OUTPUT / "full", degraded=False, request_id="phase8-8f5-smoke"
    )
    gaussian = full_fixture.store.find_unique(GaussianSceneArtifact)
    nerf = full_fixture.store.find_unique(NerfSceneArtifact)
    gaussian_preview = (Path(gaussian.artifact_dir) / "preview.png").resolve()
    nerf_preview = (Path(nerf.artifact_dir) / "preview.png").resolve()

    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    page.set_persistent_scene(full_scene, str(full_manifest))
    app.processEvents()

    resource_ready = page.appearance_resource_fallback_presentation
    cpu_static_preview_supported = bool(
        resource_ready is not None
        and "CPU" in resource_ready.resources_text
        and "static previews remain supported" in resource_ready.resources_text
        and page._gaussian_show_button.isEnabled()
    )

    _activate(page, "Gaussian", app)
    gaussian_active = bool(
        page.gaussian_preview_visible
        and page.active_view_representation == "gaussian_scene"
        and page.appearance_resource_fallback_presentation is not None
        and not page.appearance_resource_fallback_presentation.fallback_active
    )

    # Missing selected preview must fail closed to canonical, never to Gaussian.
    nerf_preview.unlink()
    page._refresh_nerf_visualization_controls()
    _activate(page, "NeRF", app)
    unavailable_fallback = page.appearance_resource_fallback_presentation
    missing_preview_retains_canonical = bool(
        page.active_view_representation == "canonical_geometry"
        and not page.gaussian_preview_visible
        and not page.nerf_preview_visible
        and unavailable_fallback is not None
        and unavailable_fallback.fallback_active
        and not unavailable_fallback.cross_representation_fallback
        and "Fallback active" in page._appearance_resource_state_value.text()
    )

    # Existing-but-undecodable image is a presentation failure and also fails closed.
    nerf_preview.write_bytes(b"not-a-decodable-image")
    page._refresh_nerf_visualization_controls()
    _activate(page, "NeRF", app)
    decode_failure_retains_canonical = bool(
        page.active_view_representation == "canonical_geometry"
        and not page.nerf_preview_visible
        and "could not be displayed" in page._appearance_resource_state_value.text()
    )

    # Restore a valid preview and prove stale failure state clears.
    _write_preview(nerf_preview, label="F5 qualified NeRF preview")
    page._refresh_nerf_visualization_controls()
    _activate(page, "NeRF", app)
    recovered = bool(
        page.nerf_preview_visible
        and page.appearance_resource_fallback_presentation is not None
        and not page.appearance_resource_fallback_presentation.fallback_active
    )

    # Persisted pipeline degradation is surfaced independently from preview display.
    degraded_scene, degraded_manifest, _ = _persistent_scene(
        OUTPUT / "degraded", degraded=True, request_id="phase8-8f5-degraded"
    )
    page.set_persistent_scene(degraded_scene, str(degraded_manifest))
    app.processEvents()
    pipeline_text = page._appearance_resource_pipeline_value.text()
    pipeline_tooltip = page._appearance_resource_pipeline_value.toolTip()
    # The persistent scene records the planner-authored stage skip reason, not
    # the fixture/provider descriptor's internal unavailable_reason. F5 must
    # validate the evidence that actually survives the persistence boundary.
    pipeline_fallback_visible = bool(
        "Gaussian appearance" in pipeline_text
        and "optional_provider_unavailable" in (pipeline_text + pipeline_tooltip)
    )
    _activate(page, "NeRF", app)

    page._readiness_badge.setText("F5 RESOURCE / FALLBACK UX • CANONICAL FAIL-CLOSED")
    window.resize(1366, 768)
    window.show()
    app.processEvents()
    page._settings_scroll.ensureWidgetVisible(page._appearance_resource_group)
    app.processEvents()
    screenshot = OUTPUT / "reconstruction_f5_resource_fallback.png"
    screenshot_ok = bool(window.grab().save(str(screenshot)))

    authority_ok = bool(
        getattr(degraded_scene, "canonical_geometry_authoritative", False)
        and page._appearance_authority_value.text()
        == "Appearance only • canonical geometry remains authoritative"
        and not bool(getattr(degraded_scene, "certified_metrology", False))
    )

    summary = {
        "phase": "8.8F5",
        "pass": bool(
            cpu_static_preview_supported
            and gaussian_active
            and missing_preview_retains_canonical
            and decode_failure_retains_canonical
            and recovered
            and pipeline_fallback_visible
            and authority_ok
            and screenshot_ok
        ),
        "resource_snapshot_source": "existing application startup hardware snapshot",
        "resource_snapshot_available": True,
        "cpu_static_preview_supported": cpu_static_preview_supported,
        "existing_static_preview_requires_external_backend": False,
        "external_gaussian_backend_executed_by_f5": False,
        "external_nerf_backend_executed_by_f5": False,
        "external_backend_resource_probe_executed_by_f5": False,
        "gaussian_active_on_cpu_snapshot": gaussian_active,
        "missing_preview_retains_canonical": missing_preview_retains_canonical,
        "decode_failure_retains_canonical": decode_failure_retains_canonical,
        "recovery_clears_stale_fallback": recovered,
        "cross_representation_fallback_executed_by_f5": False,
        "pipeline_fallback_evidence_visible": pipeline_fallback_visible,
        "canonical_geometry_authoritative": bool(getattr(degraded_scene, "canonical_geometry_authoritative", False)),
        "appearance_canonical_geometry_authoritative": False,
        "certified_metrology": False,
        "active_view": page.active_view_representation,
        "resource_ui": page._appearance_resource_resources_value.text(),
        "fallback_ui": page._appearance_resource_fallback_value.text(),
        "pipeline_ui": pipeline_text,
        "state_ui": page._appearance_resource_state_value.text(),
        "authority_ui": page._appearance_authority_value.text(),
        "screenshot": str(screenshot),
        "errors": [],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    window.close()
    app.processEvents()

    print("Phase 8.8F5 resource/fallback UX smoke: " + ("PASS" if summary["pass"] else "FAIL"))
    print(f"resources={summary['resource_ui']!r}")
    print(f"fallback={summary['fallback_ui']!r}")
    print(f"pipeline={summary['pipeline_ui']!r}")
    print(f"authority={summary['authority_ui']!r}")
    print(f"screenshot={screenshot}")
    print(f"summary={OUTPUT / 'summary.json'}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
