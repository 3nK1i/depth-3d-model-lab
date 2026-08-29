from __future__ import annotations

from copy import deepcopy

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.single_image_provider_qualification import (
    I43G4A_G1_STATUS_VALUES,
    I43G4A_MANUAL_REVIEW_DIMENSIONS,
    I43G4A_PROVIDER_IMPLEMENTATIONS,
    I43G4A_QUALITY_STATUS_VALUES,
    I43G4A_SCENARIOS,
    build_g4a_scenario_binding,
    build_i4_3g4a_preflight_snapshot,
    validate_i4_3g4a_predecessors,
    validate_runtime_probe,
)
from tests.i4_3g4a_helpers import (
    all_bindings,
    all_runtime_probes,
    g2b_snapshot,
    passing_g1_snapshot,
    passing_g3c_closure,
    runtime_probe,
)


def test_g4a_accepts_frozen_g1_and_formally_closed_g3c() -> None:
    validate_i4_3g4a_predecessors(
        i4_3g1_snapshot=passing_g1_snapshot(),
        i4_3g3c_closure=passing_g3c_closure(),
    )


def test_g4a_refuses_g1_status_schema_or_required_manual_review_drift() -> None:
    g1 = passing_g1_snapshot()
    g1["run_evidence_schema"]["status_values"].append("rejected")
    with pytest.raises(ReconstructionValidationError):
        validate_i4_3g4a_predecessors(i4_3g1_snapshot=g1, i4_3g3c_closure=passing_g3c_closure())

    g1 = passing_g1_snapshot()
    for scenario in g1["scenario_matrix"]:
        if scenario["scenario_id"] == I43G4A_SCENARIOS[0]:
            scenario["manual_review_required"] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i4_3g4a_predecessors(i4_3g1_snapshot=g1, i4_3g3c_closure=passing_g3c_closure())


def test_g4a_requires_proven_g3_rejected_quality_semantic() -> None:
    g3c = passing_g3c_closure()
    g3c["scenario_synthesis"]["g3_temporal_motion"]["production_statuses"]["rolling_depth"] = "failed"
    with pytest.raises(ReconstructionValidationError):
        validate_i4_3g4a_predecessors(i4_3g1_snapshot=passing_g1_snapshot(), i4_3g3c_closure=g3c)


def test_g4a_binds_each_g4_scenario_to_matching_g2_source_and_sfm_context() -> None:
    for scenario_id in I43G4A_SCENARIOS:
        binding = build_g4a_scenario_binding(
            scenario_id=scenario_id,
            g2b_snapshot=g2b_snapshot(scenario_id),
            g2b_snapshot_path=f"/evidence/{scenario_id}.json",
            g2b_snapshot_sha256="9" * 64,
        )
        assert binding.scenario_id == scenario_id
        assert binding.source_image_count == 12
        assert binding.registered_view_count == 8
        assert binding.landmark_count == 1024
        assert binding.authoritative_sfm_frame_id.startswith("g2b:")

    wrong = g2b_snapshot(I43G4A_SCENARIOS[0])
    with pytest.raises(ReconstructionValidationError):
        build_g4a_scenario_binding(
            scenario_id=I43G4A_SCENARIOS[1],
            g2b_snapshot=wrong,
            g2b_snapshot_path="/evidence/wrong.json",
            g2b_snapshot_sha256="9" * 64,
        )


def test_all_six_runtime_contracts_require_cuda_checkpoint_and_bound_installed_identity() -> None:
    for implementation in I43G4A_PROVIDER_IMPLEMENTATIONS:
        assert validate_runtime_probe(runtime_probe(implementation), implementation=implementation) is True
        not_ready = runtime_probe(implementation)
        not_ready["cuda_available"] = False
        assert validate_runtime_probe(not_ready, implementation=implementation) is False


def test_g4a_records_but_does_not_require_fresh_install_reproducibility_flags() -> None:
    depth_pro = runtime_probe("depth_pro")
    assert depth_pro["requirement_commit_pinned"] is False
    assert validate_runtime_probe(depth_pro, implementation="depth_pro") is True

    dav2 = runtime_probe("depth_anything_v2")
    assert dav2["installer_ref_commit_pinned"] is False
    assert dav2["checkpoint_revision_commit_pinned"] is False
    assert validate_runtime_probe(dav2, implementation="depth_anything_v2") is True

    marigold = runtime_probe("marigold")
    assert marigold["runtime_requirement_exact_pinned"] is False
    assert validate_runtime_probe(marigold, implementation="marigold") is True

    # Actual installed-runtime/source/checkpoint integrity remains mandatory.
    depth_pro["source_revision_resolved"] = False
    assert validate_runtime_probe(depth_pro, implementation="depth_pro") is False
    dav2["source_checkout_matches_marker"] = False
    assert validate_runtime_probe(dav2, implementation="depth_anything_v2") is False
    marigold["checkpoint_revision_pinned"] = False
    assert validate_runtime_probe(marigold, implementation="marigold") is False


def test_g4a_ready_snapshot_preserves_authority_and_refines_status_without_mutating_g1() -> None:
    g1 = passing_g1_snapshot()
    original_statuses = tuple(g1["run_evidence_schema"]["status_values"])
    snapshot = build_i4_3g4a_preflight_snapshot(
        i4_3g1_snapshot=g1,
        i4_3g3c_closure=passing_g3c_closure(),
        scenario_bindings=all_bindings(),
        runtime_probes=all_runtime_probes(),
        free_disk_bytes=64 * 1024**3,
        minimum_free_disk_bytes=32 * 1024**3,
    )
    assert snapshot["status"] == "ready"
    assert snapshot["pass"] is True
    assert snapshot["decision"]["i4_3g4b_production_native_execution_authorized"] is True
    assert tuple(snapshot["status_contract"]["g1_declared_status_values"]) == I43G4A_G1_STATUS_VALUES
    assert tuple(snapshot["status_contract"]["g4_quality_terminal_status_values"]) == I43G4A_QUALITY_STATUS_VALUES
    assert snapshot["status_contract"]["rejected_is_terminal_provider_quality_evidence"] is True
    assert tuple(snapshot["manual_review_contract"]["dimensions"]) == I43G4A_MANUAL_REVIEW_DIMENSIONS
    assert snapshot["manual_review_contract"]["required_for_all_three_scenarios"] is True
    assert snapshot["manual_review_contract"]["manual_review_cannot_override_production_rejection"] is True
    assert snapshot["runtime_provenance_policy"]["installed_runtime_identity_is_authoritative_for_g4a"] is True
    assert snapshot["runtime_provenance_policy"]["fresh_install_reproducibility_flags_recorded_not_readiness_gates"] is True
    assert snapshot["runtime_provenance_policy"]["depth_pro_requirement_commit_pinned_recorded"] is False
    assert snapshot["runtime_provenance_policy"]["depth_anything_v2_installer_ref_commit_pinned_recorded"] is False
    assert snapshot["runtime_provenance_policy"]["depth_anything_v2_checkpoint_requested_ref_commit_pinned_recorded"] is False
    assert snapshot["runtime_provenance_policy"]["marigold_runtime_requirement_exact_pinned_recorded"] is False
    assert snapshot["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert snapshot["authority"]["provider_selected"] is False
    assert snapshot["authority"]["permanent_global_view_cap"] is None
    assert tuple(g1["run_evidence_schema"]["status_values"]) == original_statuses


def test_g4a_not_ready_on_runtime_or_storage_failure_without_weakening_contracts() -> None:
    probes = all_runtime_probes()
    probes["marigold"] = runtime_probe("marigold", ready=False)
    snapshot = build_i4_3g4a_preflight_snapshot(
        i4_3g1_snapshot=passing_g1_snapshot(),
        i4_3g3c_closure=passing_g3c_closure(),
        scenario_bindings=all_bindings(),
        runtime_probes=probes,
        free_disk_bytes=16 * 1024**3,
        minimum_free_disk_bytes=32 * 1024**3,
    )
    assert snapshot["pass"] is False
    assert snapshot["runtime"]["marigold"]["g4a_ready"] is False
    assert snapshot["storage"]["storage_ready_for_g4b"] is False
    assert snapshot["decision"]["i4_3g4b_production_native_execution_authorized"] is False
    assert snapshot["status_contract"]["threshold_weakening_to_avoid_rejection_forbidden"] is True
