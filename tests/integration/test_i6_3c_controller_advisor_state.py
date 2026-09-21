from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.application_controller import ApplicationController
from app.reconstruction_advisor_state import ReconstructionAdvisorEvidenceBinding
from app.reconstruction_controller import ReconstructionController
from app.widgets.original_view import OriginalSourceView
from app.widgets.source_bar import SourceBar
from core.reconstruction.dense_provider_architecture import (
    DenseProviderFamily,
    DenseProviderImplementation,
)
from core.reconstruction.reconstruction_advisor_contract import (
    AdvisorContextResolution,
    AdvisorDisposition,
    AdvisorEvidenceBinding,
    AdvisorEvidenceObservation,
    AdvisorEvidenceOutcome,
    AdvisorResourceObservation,
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


def _images(root: Path) -> Path:
    path = root / "images"
    path.mkdir()
    for index in range(4):
        (path / f"frame-{index:03d}.jpg").write_bytes(b"fixture")
    return path


def _binding() -> ReconstructionAdvisorEvidenceBinding:
    evidence = AdvisorEvidenceObservation(
        provider_id="colmap-mvs-external",
        provider_family=DenseProviderFamily.MVS,
        provider_implementation=DenseProviderImplementation.COLMAP_MVS,
        outcome=AdvisorEvidenceOutcome.PASS,
        evidence_context_id="ctx-clean",
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
        evidence_context_id="ctx-clean",
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


def test_i6_3c_source_change_recomputes_passive_abstention_once(controller, tmp_path: Path) -> None:
    emitted = []
    controller.reconstruction_advisor_changed.connect(emitted.append)

    controller.select_image_sequence(_images(tmp_path))

    assert len(emitted) == 1
    assert emitted[-1].decision.disposition is AdvisorDisposition.ABSTAIN
    assert emitted[-1].decision.production_effect is False


def test_i6_3c_binding_change_emits_without_starting_or_selecting_any_provider(controller, tmp_path: Path) -> None:
    controller.select_image_sequence(_images(tmp_path))
    emitted = []
    controller.reconstruction_advisor_changed.connect(emitted.append)
    assert controller.offline_is_running is False

    controller.set_reconstruction_advisor_evidence_binding(_binding())

    assert len(emitted) == 1
    assert controller.offline_is_running is False
    assert emitted[-1].decision.disposition is AdvisorDisposition.ABSTAIN
    assert emitted[-1].decision.provider_selection_effect is False


def test_i6_3c_admitted_matching_sparse_result_recomputes_to_advisory_recommendation(
    controller, tmp_path: Path
) -> None:
    images = _images(tmp_path)
    controller.select_image_sequence(images)
    controller.set_reconstruction_advisor_evidence_binding(_binding())
    emitted = []
    controller.reconstruction_advisor_changed.connect(emitted.append)
    output = (tmp_path / "out").resolve()
    output.mkdir()

    _admit_result(controller, output, images)

    assert emitted
    state = controller.reconstruction_advisor_state
    assert state.source_matches_admitted_result is True
    assert state.decision.disposition is AdvisorDisposition.RECOMMEND
    assert state.decision.recommended_provider_id == "colmap-mvs-external"
    assert state.decision.production_effect is False
    assert controller.offline_is_running is False


def test_i6_3c_duplicate_recomputation_is_deduplicated(controller, tmp_path: Path) -> None:
    controller.select_image_sequence(_images(tmp_path))
    controller.set_reconstruction_advisor_evidence_binding(_binding())
    emitted = []
    controller.reconstruction_advisor_changed.connect(emitted.append)

    controller._emit_reconstruction_advisor_changed()
    controller._emit_reconstruction_advisor_changed()

    assert emitted == []
