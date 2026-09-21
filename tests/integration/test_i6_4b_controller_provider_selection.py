from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.application_controller import ApplicationController
from app.reconstruction_advisor_state import ReconstructionAdvisorEvidenceBinding
from app.reconstruction_controller import ReconstructionController
from app.reconstruction_provider_selection_state import ManualProviderSelectionStatus
from app.widgets.original_view import OriginalSourceView
from app.widgets.source_bar import SourceBar
from core.reconstruction.dense_provider_architecture import (
    DenseProviderFamily,
    DenseProviderImplementation,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_advisor_contract import (
    AdvisorContextResolution,
    AdvisorDisposition,
    AdvisorEvidenceBinding,
    AdvisorEvidenceObservation,
    AdvisorEvidenceOutcome,
    AdvisorResourceObservation,
)
from core.reconstruction.reconstruction_provider_selection_contract import (
    ProductionProviderSelectionMode,
)


@pytest.fixture
def controller(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    source_bar = SourceBar()
    original_view = OriginalSourceView()
    qtbot.addWidget(source_bar)
    qtbot.addWidget(original_view)
    application = ApplicationController(source_bar, original_view)
    value = ReconstructionController(application, parent=application)
    yield value
    value.shutdown()
    application.shutdown()


def _images(root: Path, name: str = "images") -> Path:
    path = root / name
    path.mkdir()
    for index in range(4):
        (path / f"frame-{index:03d}.jpg").write_bytes(b"fixture")
    return path


def _binding(context: str = "ctx-clean") -> ReconstructionAdvisorEvidenceBinding:
    evidence = AdvisorEvidenceObservation(
        provider_id="colmap-mvs-external",
        provider_family=DenseProviderFamily.MVS,
        provider_implementation=DenseProviderImplementation.COLMAP_MVS,
        outcome=AdvisorEvidenceOutcome.PASS,
        evidence_context_id=context,
        matched_scenario_ids=("scenario-clean",),
        reason_codes=("exact_clean_evidence",),
        limitations=("frozen_context_only",),
        bindings=(
            AdvisorEvidenceBinding(
                evidence_id="i5-final",
                phase="I5.14B",
                path="/evidence/i5-final.json",
                sha256="a" * 64,
            ),
        ),
    )
    return ReconstructionAdvisorEvidenceBinding(
        evidence_context_resolution=AdvisorContextResolution.EXPLICIT_FROZEN_TOKEN,
        evidence_context_id=context,
        available_provider_ids=("colmap-mvs-external",),
        resources=AdvisorResourceObservation(
            gpu_name="NVIDIA GeForce RTX 2060",
            cuda_ready=True,
            total_vram_bytes=6 * 1024**3,
            free_vram_bytes=5 * 1024**3,
            available_ram_bytes=8 * 1024**3,
        ),
        provider_evidence=(evidence,),
    )


def _admit_result(controller: ReconstructionController, root: Path, source: Path) -> None:
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


def test_i6_4b_manual_selection_records_intent_without_starting_runtime(
    controller, tmp_path: Path
) -> None:
    controller.select_image_sequence(_images(tmp_path))
    controller.set_reconstruction_advisor_evidence_binding(_binding())
    assert controller.offline_is_running is False

    state = controller.select_reconstruction_provider("colmap-mvs-external")

    assert state.status is ManualProviderSelectionStatus.SELECTED
    assert state.snapshot.selection_mode is ProductionProviderSelectionMode.MANUAL
    assert state.selected_provider_id == "colmap-mvs-external"
    assert controller.offline_is_running is False
    assert state.snapshot.provider_execution_authorized is False
    assert state.snapshot.settings_mutation_authorized is False
    assert state.snapshot.start_action_authorized is False


def test_i6_4b_advisor_recommendation_does_not_auto_select_provider(
    controller, tmp_path: Path
) -> None:
    images = _images(tmp_path)
    controller.select_image_sequence(images)
    controller.set_reconstruction_advisor_evidence_binding(_binding())
    out = tmp_path / "out"
    out.mkdir()

    _admit_result(controller, out, images)

    assert controller.reconstruction_advisor_decision.disposition is AdvisorDisposition.RECOMMEND
    state = controller.reconstruction_provider_selection_state
    assert state.recommended_provider_id == "colmap-mvs-external"
    assert state.selected_provider_id == ""
    assert state.snapshot.selection_mode is ProductionProviderSelectionMode.NONE


def test_i6_4b_same_context_passive_refresh_retains_manual_intent(
    controller, tmp_path: Path
) -> None:
    controller.select_image_sequence(_images(tmp_path))
    binding = _binding()
    controller.set_reconstruction_advisor_evidence_binding(binding)
    controller.select_reconstruction_provider("colmap-mvs-external")

    controller._emit_reconstruction_advisor_changed()
    controller._reconcile_reconstruction_provider_selection()

    state = controller.reconstruction_provider_selection_state
    assert state.status is ManualProviderSelectionStatus.SELECTED
    assert state.selected_provider_id == "colmap-mvs-external"


def test_i6_4b_source_change_fail_closed_clears_manual_intent(
    controller, tmp_path: Path
) -> None:
    controller.select_image_sequence(_images(tmp_path, "images-a"))
    controller.set_reconstruction_advisor_evidence_binding(_binding())
    controller.select_reconstruction_provider("colmap-mvs-external")

    controller.select_image_sequence(_images(tmp_path, "images-b"))

    state = controller.reconstruction_provider_selection_state
    assert state.status is ManualProviderSelectionStatus.CLEARED_INVALID
    assert state.selected_provider_id == ""
    assert "source_selection_changed" in state.reason_codes


def test_i6_4b_evidence_context_clear_fail_closed_clears_manual_intent(
    controller, tmp_path: Path
) -> None:
    controller.select_image_sequence(_images(tmp_path))
    controller.set_reconstruction_advisor_evidence_binding(_binding())
    controller.select_reconstruction_provider("colmap-mvs-external")

    controller.clear_reconstruction_advisor_evidence_binding()

    state = controller.reconstruction_provider_selection_state
    assert state.status is ManualProviderSelectionStatus.CLEARED_INVALID
    assert state.selected_provider_id == ""
    assert "selection_context_unresolved" in state.reason_codes


def test_i6_4b_clear_is_explicit_and_deduplicated(controller, tmp_path: Path) -> None:
    controller.select_image_sequence(_images(tmp_path))
    controller.set_reconstruction_advisor_evidence_binding(_binding())
    controller.select_reconstruction_provider("colmap-mvs-external")
    emitted = []
    controller.reconstruction_provider_selection_changed.connect(emitted.append)

    controller.clear_reconstruction_provider_selection()
    controller.clear_reconstruction_provider_selection()

    assert len(emitted) == 1
    assert emitted[0].status is ManualProviderSelectionStatus.NONE


def test_i6_4b_invalid_or_auto_selection_is_rejected_without_runtime_effect(
    controller, tmp_path: Path
) -> None:
    controller.select_image_sequence(_images(tmp_path))
    controller.set_reconstruction_advisor_evidence_binding(_binding())

    with pytest.raises(ReconstructionValidationError):
        controller.select_reconstruction_provider("rolling-depth-temporal")
    with pytest.raises(ReconstructionValidationError, match="Auto is not a concrete"):
        controller.select_reconstruction_provider("auto")

    assert controller.offline_is_running is False
    assert controller.reconstruction_provider_selection_state.selected_provider_id == ""
