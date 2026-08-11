#!/usr/bin/env python3
"""Decisive Phase 8.8G1 reconstruction session-model smoke."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.reconstruction_session_model import (
    ReconstructionSessionIdentityStatus,
    build_reconstruction_session_model,
)
from app.widgets.mode_tabs import AppMode

OUTPUT = Path("/tmp/phase8_8g1_session_model")
SESSION = "77777777-7777-4777-8777-777777777777"
OTHER = "88888888-8888-4888-8888-888888888888"


def _selection():
    return SimpleNamespace(
        workflow=SimpleNamespace(value="image_sequence"),
        ready=True,
        display_name="Generic reconstruction capture",
        detail="Use-case-agnostic overlapping image sequence",
        path=OUTPUT / "source",
    )


def _artifact(index: int, kind: str, authority: str, provider: str):
    return SimpleNamespace(
        artifact_id=f"g1-smoke-artifact-{index}",
        stage=SimpleNamespace(value=f"stage_{index}"),
        kind=SimpleNamespace(value=kind),
        authority=SimpleNamespace(value=authority),
        provider_id=provider,
    )


def _provider(provider_id: str, stage: str):
    return SimpleNamespace(
        provider_id=provider_id,
        provider_version="1.0",
        execution=SimpleNamespace(value="external_process"),
        stages=(SimpleNamespace(value=stage),),
    )


def _scene(session_id: str):
    return SimpleNamespace(
        scene_id="prs-g1-smoke",
        session_id=session_id,
        state=SimpleNamespace(value="ready"),
        artifacts=(
            _artifact(1, "surface_mesh", "canonical_geometry", "mesh-provider"),
            _artifact(2, "semantic_scene", "semantic_context", "semantic-provider"),
            _artifact(3, "future_visual_representation", "appearance_only", "visual-provider"),
        ),
        provider_provenance=(
            _provider("mesh-provider", "surface_mesh"),
            _provider("semantic-provider", "semantic_geometry"),
            _provider("visual-provider", "future_visual"),
        ),
        canonical_geometry_authoritative=True,
        certified_metrology=False,
        created_at_utc="2026-08-11T12:00:00+00:00",
    )


def _checkpoint(session_id: str):
    return SimpleNamespace(
        checkpoint_id="99999999-9999-4999-8999-999999999999",
        session_id=session_id,
        next_stage=SimpleNamespace(value="dense_geometry"),
        resume_allowed=True,
        provider_versions=(),
        stage_results=(),
        created_at_utc="2026-08-11T11:00:00+00:00",
    )


def main() -> int:
    shutil.rmtree(OUTPUT, ignore_errors=True)
    (OUTPUT / "source").mkdir(parents=True, exist_ok=True)

    consistent = build_reconstruction_session_model(
        source_selection=_selection(),
        scene=_scene(SESSION),
        scene_manifest=OUTPUT / "persistent_reconstruction_scene.json",
        checkpoint=_checkpoint(SESSION),
        checkpoint_manifest=OUTPUT / "reconstruction_engine_checkpoint.json",
        scene_availability=SimpleNamespace(value="ready"),
    )
    conflict = build_reconstruction_session_model(
        source_selection=_selection(),
        offline_session_id=SESSION,
        offline_state=SimpleNamespace(value="running"),
        scene=_scene(OTHER),
        checkpoint=_checkpoint(SESSION),
    )

    CameraEnumerator.start = lambda self: None  # type: ignore[method-assign]
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page

    page.set_session_model(conflict)
    app.processEvents()
    conflict_visible = bool(
        page._session_context_value.text() == "Session identity conflict"
        and page._session_context_value.property("statusKind") == "warning"
        and OTHER in page._session_value.toolTip()
    )

    page.set_session_model(consistent)
    app.processEvents()
    recovered = bool(
        page._session_value.text() == SESSION[:8]
        and page._session_context_value.text() == "Persistent scene • Ready"
        and "3 artifacts" in page._session_evidence_value.text()
        and "3 providers" in page._session_evidence_value.text()
        and "future_visual_representation" in page._session_evidence_value.toolTip()
        and page._scene_manifest_value.text() == "persistent_reconstruction_scene.json"
        and page._checkpoint_value.text() == "reconstruction_engine_checkpoint.json"
    )

    window.resize(1366, 768)
    window.show()
    app.processEvents()
    screenshot = OUTPUT / "reconstruction_g1_session_model.png"
    screenshot_ok = window.grab().save(str(screenshot))

    summary = {
        "phase": "8.8G1",
        "pass": bool(
            consistent.identity_status is ReconstructionSessionIdentityStatus.CONSISTENT
            and conflict.identity_status is ReconstructionSessionIdentityStatus.CONFLICT
            and conflict_visible
            and recovered
            and consistent.canonical_geometry_authoritative
            and not consistent.certified_metrology
            and screenshot_ok
        ),
        "session_id": consistent.session_id,
        "focus": consistent.focus.value,
        "lifecycle": consistent.lifecycle.value,
        "identity_consistent": consistent.identity_status.value == "consistent",
        "identity_conflict_detected": conflict.identity_status.value == "conflict",
        "identity_conflict_visible": conflict_visible,
        "identity_recovery_visible": recovered,
        "artifact_count": len(consistent.artifacts),
        "provider_count": len(consistent.providers),
        "future_representation_supported_generically": "future_visual_representation" in consistent.appearance_representation_kinds,
        "canonical_geometry_authoritative": consistent.canonical_geometry_authoritative,
        "certified_metrology": False,
        "new_persistence_format_written_by_g1": False,
        "save_reload_executed_by_g1": False,
        "checkpoint_resume_executed_by_g1": False,
        "failure_recovery_executed_by_g1": False,
        "session_ui": page._session_value.text(),
        "context_ui": page._session_context_value.text(),
        "evidence_ui": page._session_evidence_value.text(),
        "screenshot": str(screenshot),
        "errors": [],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    window.close()
    app.processEvents()

    print("Phase 8.8G1 reconstruction session model smoke: " + ("PASS" if summary["pass"] else "FAIL"))
    print(f"session={summary['session_id']!r} context={summary['context_ui']!r}")
    print(f"evidence={summary['evidence_ui']!r}")
    print(f"screenshot={screenshot}")
    print(f"summary={OUTPUT / 'summary.json'}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
