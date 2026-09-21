from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.main_window import MainWindow
from app.reconstruction_advisor_state import ReconstructionAdvisorEvidenceBinding
from core.reconstruction.dense_provider_architecture import (
    DenseProviderFamily,
    DenseProviderImplementation,
)
from core.reconstruction.reconstruction_advisor_contract import (
    AdvisorContextResolution,
    AdvisorEvidenceBinding,
    AdvisorEvidenceObservation,
    AdvisorEvidenceOutcome,
    AdvisorResourceObservation,
)

SHA = "a" * 64


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()



def _settings_group_titles(page):
    layout = page._settings_scroll.widget().layout()
    titles = []
    for index in range(layout.count()):
        widget = layout.itemAt(index).widget()
        if hasattr(widget, "title"):
            titles.append(widget.title)
    return tuple(titles)

def _images(root: Path, name: str = "images") -> Path:
    path = root / name
    path.mkdir()
    for index in range(4):
        (path / f"frame-{index:03d}.jpg").write_bytes(b"fixture")
    return path


def _resources() -> AdvisorResourceObservation:
    return AdvisorResourceObservation(
        gpu_name="NVIDIA GeForce RTX 2060",
        cuda_ready=True,
        total_vram_bytes=6 * 1024**3,
        free_vram_bytes=5 * 1024**3,
        available_ram_bytes=8 * 1024**3,
    )


def _observation(
    provider_id: str,
    implementation: DenseProviderImplementation,
    family: DenseProviderFamily,
) -> AdvisorEvidenceObservation:
    return AdvisorEvidenceObservation(
        provider_id=provider_id,
        provider_family=family,
        provider_implementation=implementation,
        outcome=AdvisorEvidenceOutcome.PASS,
        evidence_context_id="ctx-clean",
        matched_scenario_ids=(f"scenario-{provider_id}",),
        reason_codes=("exact_clean_evidence",),
        limitations=("frozen_context_only",),
        bindings=(
            AdvisorEvidenceBinding(
                evidence_id=f"evidence-{provider_id}",
                phase="I5.14B",
                path=f"/evidence/{provider_id}.json",
                sha256=SHA,
            ),
        ),
    )


def _binding(*, ambiguous: bool = False) -> ReconstructionAdvisorEvidenceBinding:
    evidence = [
        _observation(
            "colmap-mvs-external",
            DenseProviderImplementation.COLMAP_MVS,
            DenseProviderFamily.MVS,
        )
    ]
    providers = ["colmap-mvs-external"]
    if ambiguous:
        evidence.append(
            _observation(
                "depth-pro",
                DenseProviderImplementation.DEPTH_PRO,
                DenseProviderFamily.SINGLE_IMAGE_DEPTH,
            )
        )
        providers.append("depth-pro")
    return ReconstructionAdvisorEvidenceBinding(
        evidence_context_resolution=AdvisorContextResolution.EXPLICIT_FROZEN_TOKEN,
        evidence_context_id="ctx-clean",
        available_provider_ids=tuple(providers),
        resources=_resources(),
        provider_evidence=tuple(evidence),
    )


def _admit_result(window: MainWindow, root: Path, source: Path) -> None:
    controller = window._reconstruction_controller
    sid = str(uuid4())
    controller.offline_controller._session_id = sid
    controller.offline_controller._output_dir = root
    payload = {
        "session_id": sid,
        "status": "completed",
        "pass": True,
        "camera_pose_available": True,
        "canonical_geometry_authoritative": False,
        "persistent_scene_ready": False,
        "certified_metrology": False,
        "source": {"source_uri": str(source.resolve()), "fingerprint": "fp"},
        "sfm_summary": {
            "quality_assessment": {"status": "accepted", "usable": True},
            "gauge": {"scale_status": "arbitrary"},
        },
        "artifacts": {},
    }
    (root / "offline_result.json").write_text(json.dumps(payload), encoding="utf-8")
    controller._on_offline_result_ready(payload, str(root))


def test_initial_advisor_ui_is_clear_non_error_abstention(window) -> None:
    page = window._reconstruction_page

    assert page._advisor_status_value.text() == "No recommendation"
    assert page._advisor_status_value.property("statusKind") == "neutral"
    assert page._advisor_guidance_value.text() == "Waiting for source"
    assert page._advisor_boundary_value.text() == "Advisory only • no settings changed"


def test_ready_source_waits_for_authoritative_sparse_sfm(window, tmp_path: Path) -> None:
    controller = window._reconstruction_controller
    controller.select_image_sequence(_images(tmp_path))
    page = window._reconstruction_page

    assert page._advisor_status_value.text() == "No recommendation"
    assert page._advisor_status_value.property("statusKind") == "neutral"
    assert page._advisor_guidance_value.text() == "Waiting for qualified sparse SfM"


def test_exact_evidence_recommendation_is_visible_but_non_executable(window, tmp_path: Path) -> None:
    controller = window._reconstruction_controller
    source = _images(tmp_path)
    controller.select_image_sequence(source)
    controller.set_reconstruction_advisor_evidence_binding(_binding())
    output = (tmp_path / "run").resolve()
    output.mkdir()

    _admit_result(window, output, source)
    page = window._reconstruction_page

    assert page._advisor_status_value.text() == "Recommendation available"
    assert page._advisor_status_value.property("statusKind") == "ready"
    assert page._advisor_guidance_value.text() == "COLMAP MVS"
    assert page._advisor_evidence_value.text() == "Exact qualified evidence"
    assert page._advisor_boundary_value.text() == "Advisory only • no settings changed"
    assert controller.offline_is_running is False


def test_source_change_invalidates_previous_recommendation_visibly(window, tmp_path: Path) -> None:
    controller = window._reconstruction_controller
    source = _images(tmp_path, "images-a")
    controller.select_image_sequence(source)
    controller.set_reconstruction_advisor_evidence_binding(_binding())
    output = (tmp_path / "run").resolve()
    output.mkdir()
    _admit_result(window, output, source)
    assert window._reconstruction_page._advisor_status_value.text() == "Recommendation available"

    controller.select_image_sequence(_images(tmp_path, "images-b"))
    page = window._reconstruction_page

    assert page._advisor_status_value.text() == "No recommendation"
    assert page._advisor_guidance_value.text() == "Waiting for qualified sparse SfM"
    assert page._advisor_status_value.property("statusKind") == "neutral"


def test_ambiguous_exact_evidence_is_warning_abstention_without_side_effect(window, tmp_path: Path) -> None:
    controller = window._reconstruction_controller
    source = _images(tmp_path)
    controller.select_image_sequence(source)
    controller.set_reconstruction_advisor_evidence_binding(_binding(ambiguous=True))
    output = (tmp_path / "run").resolve()
    output.mkdir()

    _admit_result(window, output, source)
    page = window._reconstruction_page

    assert page._advisor_status_value.text() == "No recommendation"
    assert page._advisor_status_value.property("statusKind") == "warning"
    assert page._advisor_guidance_value.text() == "Advisor abstained"
    assert page._advisor_evidence_value.text() == "Conflicting evidence"
    assert controller.offline_is_running is False


def test_advisor_group_is_promoted_before_dense_details_without_breaking_e4_order(window) -> None:
    page = window._reconstruction_page
    titles = _settings_group_titles(page)
    assert titles[:6] == (
        "DIGITAL TWIN WORKSPACE",
        "BUILD & PROCESSING",
        "CANONICAL SCENE",
        "RULER METRIC ESTIMATE",
        "RECONSTRUCTION ADVISOR",
        "DENSE RGB POINTS",
    )
