#!/usr/bin/env python3
"""Phase 8.9F6 actual-application end-to-end acceptance.

Loads the real self-contained F5 PersistentReconstructionScene into the actual
PyQt MainWindow, exercises persisted Canonical/Gaussian/NeRF presentation
switching and UI fallback, verifies reload identity/integrity, 1366x768
responsiveness, authority truth, and process/runtime cleanup.

F6 does not start PyCOLMAP, DA3, Gaussian training/rendering, or Nerfstudio.
The Gaussian/NeRF views are the already-authored static previews persisted by
F5. Canonical geometry remains authoritative throughout.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import monotonic, sleep
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASE = "8.9F6"
EXPECTED_BRANCH = "phase8-reconstruction"
EXPECTED_HEAD = "eb5062bebeb146ad98d658c19fab7ec4750f2834"
DEFAULT_F5_ROOT = Path("/tmp/phase8_9f5_persistent_scene")
DEFAULT_OUTPUT = Path("/tmp/phase8_9f6_actual_application")

F6_REPO_FILES = {
    "docs/phase_history/PHASE8_9F6_VALIDATION.md",
    "scripts/run_phase8_9f6_actual_application_acceptance.py",
    "scripts/run_phase8_9f6_tests.sh",
    "tests/integration/test_phase8_9f6_actual_application_contract.py",
    "tests/unit/test_phase8_9f6_actual_application_acceptance.py",
}
HEAVY_MODULES = (
    "depth_anything_3",
    "diff_gaussian_rasterization",
    "gsplat",
    "nerfstudio",
    "pycolmap",
    "simple_knn",
    "tinycudann",
)


def _git(command: list[str], *, cwd: Path = ROOT) -> str:
    completed = subprocess.run(
        ["git", *command], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "git command failed")
    return completed.stdout.strip()


def _repository_state(expected_head: str = EXPECTED_HEAD) -> dict[str, Any]:
    branch = _git(["branch", "--show-current"])
    head = _git(["rev-parse", "HEAD"])
    origin = _git(["rev-parse", "origin/phase8-reconstruction"])
    lines = _git(["status", "--short", "--untracked-files=all"]).splitlines()
    overlay: list[str] = []
    unexpected: list[str] = []
    for line in lines:
        path = line[3:].strip() if len(line) >= 4 else line.strip()
        if path in F6_REPO_FILES:
            overlay.append(path)
        else:
            unexpected.append(line)
    return {
        "branch": branch,
        "head": head,
        "origin_head": origin,
        "expected_head": expected_head,
        "head_matches_origin": head == origin,
        "head_matches_expected": head == expected_head,
        "f6_overlay_files_present": sorted(overlay),
        "unexpected_worktree_changes": unexpected,
        "pass": bool(
            branch == EXPECTED_BRANCH
            and head == origin
            and head == expected_head
            and not unexpected
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid or missing JSON evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON evidence is not an object: {path}")
    return payload


def _validate_f5_summary(payload: dict[str, Any], f5_root: Path) -> dict[str, Any]:
    authority = payload.get("authority") if isinstance(payload.get("authority"), dict) else {}
    scene = payload.get("scene") if isinstance(payload.get("scene"), dict) else {}
    readiness = payload.get("presentation_readiness") if isinstance(payload.get("presentation_readiness"), dict) else {}
    gaussian = readiness.get("gaussian") if isinstance(readiness.get("gaussian"), dict) else {}
    nerf = readiness.get("nerf") if isinstance(readiness.get("nerf"), dict) else {}
    checks = {
        "phase": payload.get("phase") == "8.9F5",
        "pass": payload.get("pass") is True,
        "no_blockers": payload.get("blockers") == [],
        "canonical_authoritative": authority.get("canonical_geometry") == "authoritative",
        "gaussian_appearance_only": authority.get("gaussian") == "appearance_only",
        "nerf_appearance_only": authority.get("nerf") == "appearance_only",
        "not_certified_metrology": authority.get("certified_metrology") is False,
        "scene_ready": scene.get("state") == "ready",
        "scene_canonical_authoritative": scene.get("canonical_geometry_authoritative") is True,
        "gaussian_preview_ready": gaussian.get("ready") is True,
        "nerf_preview_ready": nerf.get("ready") is True,
    }
    manifest = Path(str(scene.get("manifest", ""))).expanduser()
    if not manifest.is_absolute():
        manifest = f5_root / "scene-a" / "persistent_reconstruction_scene.json"
    manifest = manifest.resolve()
    checks["manifest_exists"] = manifest.is_file()
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError(f"F5 source qualification failed: {failed}")
    return {"checks": checks, "manifest": manifest, "scene": scene, "readiness": readiness}


def _heavy_importability() -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in HEAVY_MODULES}


def _heavy_loaded() -> list[str]:
    loaded: list[str] = []
    for name in HEAVY_MODULES:
        if any(module == name or module.startswith(name + ".") for module in sys.modules):
            loaded.append(name)
    return sorted(loaded)


def _child_pids() -> list[int]:
    completed = subprocess.run(
        ["ps", "-o", "pid=", "--ppid", str(os.getpid())],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    if completed.returncode not in (0, 1):
        return []
    result: list[int] = []
    for token in completed.stdout.split():
        try:
            pid = int(token)
        except ValueError:
            continue
        if pid > 0:
            result.append(pid)
    return sorted(result)


def _pump(app: Any, seconds: float = 0.05) -> None:
    deadline = monotonic() + max(0.0, float(seconds))
    while monotonic() < deadline:
        app.processEvents()
        sleep(0.01)
    app.processEvents()


def _save_window(window: Any, path: Path) -> None:
    pixmap = window.grab()
    if pixmap.isNull() or not pixmap.save(str(path)):
        raise RuntimeError(f"could not save F6 application screenshot: {path}")


def _sidebar_metrics(page: Any) -> dict[str, Any]:
    from PyQt6.QtCore import Qt

    scroll = page._settings_scroll
    body = scroll.widget()
    hbar = scroll.horizontalScrollBar()
    return {
        "scroll_width": scroll.width(),
        "viewport_width": scroll.viewport().width(),
        "body_width": 0 if body is None else body.width(),
        "horizontal_maximum": hbar.maximum(),
        "horizontal_visible": hbar.isVisible(),
        "horizontal_policy_always_off": (
            scroll.horizontalScrollBarPolicy() is Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        ),
    }


def _activate_label(page: Any, label: str, app: Any) -> int:
    index = page._appearance_representation_combo.findText(label)
    if index < 0:
        raise RuntimeError(f"appearance selector does not expose {label!r}")
    page._appearance_representation_combo.setCurrentIndex(index)
    page._appearance_representation_combo.activated.emit(index)
    _pump(app, 0.08)
    return index


def _fallback_scene_clone(scene: Any, missing_dir: Path) -> Any:
    """Create an in-memory UI-only clone with an unavailable NeRF preview.

    The real F5 scene/artifacts are never modified. This probe targets the
    existing application fallback presentation only; F5 already qualified
    persistent-file tamper/missing rejection separately.
    """

    modified = []
    gaussian_artifact = None
    nerf_artifact = None
    for artifact in tuple(scene.artifacts):
        if str(getattr(getattr(artifact, "kind", ""), "value", getattr(artifact, "kind", ""))) == "nerf_scene":
            raw = artifact.as_dict()
            snapshot = dict(raw.get("payload_snapshot") or {})
            snapshot["artifact_dir"] = str(missing_dir.resolve())
            snapshot["preview_path"] = str((missing_dir / "missing-preview.png").resolve())
            replacement = replace(artifact, payload_snapshot=snapshot)
            modified.append(replacement)
            nerf_artifact = replacement
        else:
            modified.append(artifact)
            if str(getattr(getattr(artifact, "kind", ""), "value", getattr(artifact, "kind", ""))) == "gaussian_scene":
                gaussian_artifact = artifact
    missing_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        scene_id=scene.scene_id,
        request_id=scene.request_id,
        session_id=scene.session_id,
        state=scene.state,
        pipeline_status=scene.pipeline_status,
        artifacts=tuple(modified),
        provider_provenance=scene.provider_provenance,
        fallback_evidence=scene.fallback_evidence,
        file_references=scene.file_references,
        canonical_geometry_artifact_id=scene.canonical_geometry_artifact_id,
        scene_fingerprint_sha256=scene.scene_fingerprint_sha256,
        created_at_utc=scene.created_at_utc,
        certified_metrology=False,
        reconstruction_ui_enabled=False,
        canonical_geometry_authoritative=True,
        gaussian_artifact=gaussian_artifact,
        nerf_artifact=nerf_artifact,
    )


def _make_contact_sheet(paths: list[Path], output: Path) -> None:
    try:
        from PIL import Image, ImageOps, ImageDraw
    except ImportError:
        return
    images = [Image.open(path).convert("RGB") for path in paths if path.is_file()]
    if not images:
        return
    width = max(image.width for image in images)
    label_height = 34
    height = sum(image.height + label_height for image in images)
    sheet = Image.new("RGB", (width, height), (14, 16, 20))
    draw = ImageDraw.Draw(sheet)
    y = 0
    labels = ["CANONICAL / SCENE READY", "GAUSSIAN / APPEARANCE ONLY", "NERF / APPEARANCE ONLY", "FALLBACK / CANONICAL RETAINED"]
    for index, image in enumerate(images):
        padded = ImageOps.pad(image, (width, image.height))
        draw.text((12, y + 8), labels[index] if index < len(labels) else f"VIEW {index + 1}", fill=(230, 230, 230))
        y += label_height
        sheet.paste(padded, (0, y))
        y += image.height
    sheet.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f5-root", type=Path, default=DEFAULT_F5_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hold-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if args.hold_seconds < 0:
        parser.error("--hold-seconds cannot be negative")

    f5_root = args.f5_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    f5_summary_path = f5_root / "summary.json"
    f5_summary = _load_json(f5_summary_path)
    f5_evidence = _validate_f5_summary(f5_summary, f5_root)
    repository = _repository_state()
    if not repository["pass"]:
        raise RuntimeError(f"repository baseline/scope is not clean for F6: {repository}")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    # This is an actual desktop UI acceptance. Never inherit prior fake/offscreen
    # visualization flags. We do not force a platform; the user's desktop session
    # remains authoritative for the release screenshot.
    os.environ.pop("DEPTH3D_TEST_FAKE_VISPY", None)
    os.environ.pop("QT_QPA_PLATFORM", None)
    os.environ.pop("QT_OPENGL", None)
    os.environ.pop("LIBGL_ALWAYS_SOFTWARE", None)

    from PyQt6.QtWidgets import QApplication
    from app.application_controller import CameraEnumerator
    from app.main_window import MainWindow
    from app.widgets.mode_tabs import AppMode
    from core.reconstruction import verify_persistent_scene_file_references

    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = MainWindow()
    window.resize(1366, 768)
    window.show()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    _pump(app, 0.2)

    page = window._reconstruction_page
    controller = window._reconstruction_controller
    errors: list[str] = []
    controller.error_occurred.connect(errors.append)
    screenshots: list[Path] = []
    heavy_before = _heavy_loaded()
    children_before = _child_pids()
    summary: dict[str, Any] = {
        "phase": PHASE,
        "pass": False,
        "blockers": [],
        "errors": errors,
        "repository": repository,
        "production_code_changed_by_8_9f6": False,
        "new_ui_architecture_created": False,
        "heavy_backends_started": False,
        "certified_metrology": False,
    }

    try:
        manifest = f5_evidence["manifest"]
        scene = controller.load_persistent_scene(manifest, verify_file_references=True)
        _pump(app, 0.15)
        verify_persistent_scene_file_references(scene)

        original_id = scene.scene_id
        original_fingerprint = scene.scene_fingerprint_sha256
        gaussian_expected = Path(str(f5_summary["materialized_artifacts"]["gaussian"]["preview_path"])).resolve()
        nerf_expected = Path(str(f5_summary["materialized_artifacts"]["nerf"]["preview_path"])).resolve()
        canonical_expected = Path(str(f5_summary["materialized_artifacts"]["canonical"]["surface_obj_path"])).resolve()

        for group in (
            page._session_group,
            page._appearance_group,
            page._gaussian_visualization_group,
            page._nerf_visualization_group,
            page._appearance_resource_group,
            page._authority_group,
        ):
            group.set_expanded(True)
        page._settings_scroll.ensureWidgetVisible(page._appearance_group)
        _pump(app, 0.12)

        canonical_loaded = bool(
            controller.persistent_scene is scene
            and controller.persistent_scene_manifest == manifest.resolve()
            and scene.canonical_geometry_authoritative
            and scene.certified_metrology is False
            and canonical_expected.is_file()
            and page.active_view_representation == "canonical_geometry"
            and not page.gaussian_preview_visible
            and not page.nerf_preview_visible
            and page._viewport_state.text() == "Scene ready"
        )
        canonical_png = output / "01_canonical_scene_ready.png"
        _save_window(window, canonical_png)
        screenshots.append(canonical_png)

        gaussian_index = _activate_label(page, "Gaussian", app)
        gaussian_active = bool(
            gaussian_index >= 0
            and page.gaussian_preview_visible
            and not page.nerf_preview_visible
            and page.active_view_representation == "gaussian_scene"
            and Path(page._appearance_preview.source_path).resolve() == gaussian_expected
            and page._appearance_authority_value.text() == "Appearance only • canonical geometry remains authoritative"
            and page._appearance_resource_state_value.text() == "Appearance view active • canonical authority retained"
        )
        page._settings_scroll.ensureWidgetVisible(page._gaussian_visualization_group)
        _pump(app, 0.08)
        gaussian_png = output / "02_gaussian_appearance.png"
        _save_window(window, gaussian_png)
        screenshots.append(gaussian_png)

        nerf_index = _activate_label(page, "NeRF", app)
        nerf_active = bool(
            nerf_index >= 0
            and not page.gaussian_preview_visible
            and page.nerf_preview_visible
            and page.active_view_representation == "nerf_scene"
            and Path(page._appearance_preview.source_path).resolve() == nerf_expected
            and page._appearance_authority_value.text() == "Appearance only • canonical geometry remains authoritative"
            and page._appearance_resource_state_value.text() == "Appearance view active • canonical authority retained"
        )
        page._settings_scroll.ensureWidgetVisible(page._nerf_visualization_group)
        _pump(app, 0.08)
        nerf_png = output / "03_nerf_appearance.png"
        _save_window(window, nerf_png)
        screenshots.append(nerf_png)

        _activate_label(page, "Gaussian", app)
        page._gaussian_hide_button.click()
        _pump(app, 0.08)
        hide_returns_canonical = bool(
            page.active_view_representation == "canonical_geometry"
            and not page.gaussian_preview_visible
            and not page.nerf_preview_visible
        )

        reloaded = controller.reload_persistent_scene()
        _pump(app, 0.12)
        reload_identity = bool(
            reloaded.scene_id == original_id
            and reloaded.scene_fingerprint_sha256 == original_fingerprint
            and controller.persistent_scene_manifest == manifest.resolve()
            and reloaded.canonical_geometry_authoritative
            and reloaded.certified_metrology is False
            and page.active_view_representation == "canonical_geometry"
        )

        # UI-only fallback probe from an in-memory clone. The real F5 manifest
        # and all six referenced files remain untouched and are re-verified after.
        fallback_scene = _fallback_scene_clone(reloaded, output / "missing-nerf-preview-probe")
        page.set_persistent_scene(fallback_scene, str(manifest))
        _pump(app, 0.08)
        _activate_label(page, "NeRF", app)
        fallback = page.appearance_resource_fallback_presentation
        fallback_ok = bool(
            page.active_view_representation == "canonical_geometry"
            and not page.gaussian_preview_visible
            and not page.nerf_preview_visible
            and fallback is not None
            and fallback.fallback_active
            and not fallback.cross_representation_fallback
            and page._appearance_resource_fallback_value.text() == "Canonical geometry • no neural cross-fallback"
            and "NeRF" in page._appearance_resource_state_value.text()
        )
        page._settings_scroll.ensureWidgetVisible(page._appearance_resource_group)
        _pump(app, 0.08)
        fallback_png = output / "04_missing_nerf_fallback.png"
        _save_window(window, fallback_png)
        screenshots.append(fallback_png)

        # Restore the exact controller-owned real scene and prove appearance is
        # still usable after the fallback probe.
        page.set_persistent_scene(reloaded, str(manifest))
        _pump(app, 0.08)
        _activate_label(page, "NeRF", app)
        recovery_after_fallback = bool(page.nerf_preview_visible and page.active_view_representation == "nerf_scene")
        page._nerf_hide_button.click()
        _pump(app, 0.05)
        verify_persistent_scene_file_references(reloaded)

        sidebar = _sidebar_metrics(page)
        responsive = bool(
            window.width() == 1366
            and window.height() == 768
            and sidebar["horizontal_maximum"] == 0
            and sidebar["horizontal_visible"] is False
            and sidebar["horizontal_policy_always_off"] is True
        )
        authority_ok = bool(
            reloaded.canonical_geometry_authoritative
            and reloaded.certified_metrology is False
            and page._appearance_authority_value.text() == "Appearance only • canonical geometry remains authoritative"
        )
        presentation_only = bool(
            gaussian_expected.is_file()
            and nerf_expected.is_file()
            and not any(_heavy_importability().values())
            and not _heavy_loaded()
        )

        contact_sheet = output / "f6_application_acceptance_contact_sheet.png"
        _make_contact_sheet(screenshots, contact_sheet)

        checks = {
            "actual_main_window": True,
            "reconstruction_mode_active": window._mode_tabs.active_mode is AppMode.RECONSTRUCTION,
            "real_f5_scene_loaded_through_controller": canonical_loaded,
            "canonical_scene_ready_state": canonical_loaded,
            "gaussian_switch": gaussian_active,
            "nerf_switch": nerf_active,
            "hide_returns_canonical": hide_returns_canonical,
            "reload_identity_stable": reload_identity,
            "missing_nerf_falls_back_to_canonical": fallback_ok,
            "fallback_has_no_neural_cross_substitution": fallback_ok,
            "recovery_after_fallback": recovery_after_fallback,
            "authority_preserved": authority_ok,
            "persisted_preview_only_no_backend_runtime": presentation_only,
            "responsive_1366x768": responsive,
            "f5_references_intact_after_ui_probe": True,
            "no_controller_errors": not errors,
        }
        blockers = sorted(key for key, value in checks.items() if not value)
        summary.update(
            {
                "pass": not blockers,
                "blockers": blockers,
                "checks": checks,
                "source_f5": {
                    "summary": str(f5_summary_path),
                    "manifest": str(manifest),
                    "scene_id": original_id,
                    "scene_fingerprint_sha256": original_fingerprint,
                    "canonical_surface_obj": str(canonical_expected),
                    "gaussian_preview": str(gaussian_expected),
                    "nerf_preview": str(nerf_expected),
                },
                "authority": {
                    "canonical_geometry": "authoritative",
                    "gaussian": "appearance_only",
                    "nerf": "appearance_only",
                    "certified_metrology": False,
                },
                "application": {
                    "window_size": [window.width(), window.height()],
                    "active_mode": "reconstruction",
                    "initial_view": "canonical_geometry",
                    "canonical_view_scope": "scene_ready_authority_state; F6 does not add a persisted-OBJ mesh renderer",
                    "gaussian_view_scope": "persisted_backend_authored_static_preview",
                    "nerf_view_scope": "persisted_backend_authored_static_preview",
                    "sidebar": sidebar,
                    "screenshots": [str(path) for path in screenshots],
                    "contact_sheet": str(contact_sheet) if contact_sheet.is_file() else "",
                },
                "fallback_probe": {
                    "scope": "in_memory_ui_presentation_clone_no_f5_artifact_mutation",
                    "selected": "nerf_scene",
                    "fallback_target": "canonical_geometry",
                    "cross_representation_fallback": False,
                    "passed": fallback_ok,
                },
                "isolation": {
                    "heavy_module_importability_in_depth": _heavy_importability(),
                    "heavy_modules_loaded_before": heavy_before,
                    "heavy_modules_loaded_during": _heavy_loaded(),
                    "production_heavy_runtime_leak": bool(_heavy_loaded()),
                    "heavy_backend_training_or_inference_started": False,
                },
                "real_execution": {
                    "actual_pyqt_main_window": True,
                    "f5_persistent_scene_loaded": True,
                    "persistent_scene_reloaded": True,
                    "gaussian_static_preview_displayed": gaussian_active,
                    "nerf_static_preview_displayed": nerf_active,
                    "pycolmap_started": False,
                    "da3_started": False,
                    "gaussian_training_started": False,
                    "nerfacto_training_started": False,
                },
                "truthful_scope": {
                    "canonical_geometry_authoritative": True,
                    "appearance_promoted_to_geometry_authority": False,
                    "persisted_canonical_obj_directly_rendered_by_new_f6_code": False,
                    "new_application_rendering_architecture_created": False,
                    "statement": (
                        "F6 loaded the real self-contained F5 scene through the actual ReconstructionController/MainWindow, "
                        "proved canonical scene-ready authority state plus persisted Gaussian/NeRF static-preview switching, "
                        "reload identity, fail-closed canonical fallback, and 1366x768 UI behavior. No heavy backend was launched."
                    ),
                },
            }
        )
        if args.hold_seconds:
            _pump(app, args.hold_seconds)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        summary["blockers"] = sorted(set(list(summary.get("blockers", [])) + ["campaign_exception"]))
        summary["pass"] = False
    finally:
        try:
            window.close()
            _pump(app, 0.15)
        except Exception as exc:  # pragma: no cover - defensive shutdown reporting
            errors.append(f"shutdown:{type(exc).__name__}: {exc}")

    children_after = _child_pids()
    heavy_after = _heavy_loaded()
    cleanup = {
        "window_closed": not window.isVisible(),
        "controller_active_after_close": bool(getattr(controller, "is_active", False)),
        "child_pids_before": children_before,
        "child_pids_after": children_after,
        "heavy_modules_loaded_after_close": heavy_after,
        "pass": bool(not window.isVisible() and not getattr(controller, "is_active", False) and not heavy_after),
    }
    summary["cleanup"] = cleanup
    if not cleanup["pass"]:
        summary["pass"] = False
        summary["blockers"] = sorted(set(list(summary.get("blockers", [])) + ["cleanup_failed"]))

    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Phase 8.9F6 actual application end-to-end acceptance: {'PASS' if summary.get('pass') else 'FAIL'}")
    if summary.get("pass"):
        print("canonical=PASS gaussian=PASS nerf=PASS reload=PASS fallback=PASS responsive=PASS cleanup=PASS")
        print("canonical_authoritative=True gaussian_authority=appearance_only nerf_authority=appearance_only certified_metrology=False")
        print("heavy_backends_started=False")
    else:
        print(f"blockers={summary.get('blockers', [])}")
        if errors:
            print(f"errors={errors}")
    print(f"contact_sheet={summary.get('application', {}).get('contact_sheet', '')}")
    print(f"summary={summary_path}")
    return 0 if summary.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
