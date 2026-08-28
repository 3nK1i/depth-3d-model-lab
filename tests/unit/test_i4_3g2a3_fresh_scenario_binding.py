from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_da3_fresh_scenario_binding import (
    DiagnosticFrame,
    FreshBindingPolicy,
    LEGACY_LOW_TEXTURE_METHOD,
    LOW_TEXTURE_AMENDMENT_ID,
    LOW_TEXTURE_METHOD,
    _low_texture_component_diagnostic,
    build_fresh_scenario_binding_snapshot,
    validate_fresh_scenario_binding_snapshot,
)
from tests.i4_3g2a3_helpers import make_multiview_fixture


def _ready_snapshot(tmp_path: Path) -> dict:
    source = make_multiview_fixture(tmp_path / "room")
    return build_fresh_scenario_binding_snapshot(
        complete_room_dir=source,
        output_dir=tmp_path / "binding",
        confirm_complete_room=True,
        confirm_low_texture=True,
        confirm_weak_overlap=True,
    )


def test_policy_is_explicit_and_pre_provider() -> None:
    policy = FreshBindingPolicy().as_dict()
    assert policy["low_texture_max_median_ratio_to_reference"] == 0.75
    assert policy["low_texture_max_component_ratio_to_reference"] == 0.75
    assert policy["weak_overlap_minimum_stride"] == 2
    assert policy["weak_overlap_min_good_matches_per_pair"] == 8


def test_unconfirmed_semantics_do_not_authorize(tmp_path: Path) -> None:
    source = make_multiview_fixture(tmp_path / "room")
    snapshot = build_fresh_scenario_binding_snapshot(
        complete_room_dir=source,
        output_dir=tmp_path / "binding",
        confirm_complete_room=False,
        confirm_low_texture=False,
        confirm_weak_overlap=False,
    )
    assert snapshot["pass"] is True
    assert snapshot["status"] == "needs_confirmation_or_qualification"
    assert snapshot["binding_summary"]["all_fresh_bindings_ready"] is False
    assert snapshot["decision"]["g2b_authorized_by_this_artifact_alone"] is False
    assert snapshot["scope"]["real_provider_execution_started"] is False


def test_controlled_derivatives_can_be_frozen_when_confirmed(tmp_path: Path) -> None:
    snapshot = _ready_snapshot(tmp_path)
    assert snapshot["status"] == "ready"
    assert snapshot["binding_summary"]["ready_fresh_bindings"] == 3
    by_id = {item["scenario_id"]: item for item in snapshot["fresh_bindings"]}
    low = by_id["g2_low_texture_mvs_da3"]
    weak = by_id["g2_weak_overlap_mvs_da3"]
    assert low["diagnostics"]["objective_proxy_qualified"] is True
    assert low["diagnostics"]["methodology"] == LOW_TEXTURE_METHOD
    assert low["diagnostics"]["methodology_amendment_id"] == LOW_TEXTURE_AMENDMENT_ID
    assert low["diagnostics"]["qualification_margin_ratio"] <= 0.75
    assert low["diagnostics"]["legacy_texture_proxy"]["authoritative_for_qualification"] is False
    assert low["derivation"]["mode"] == "controlled_natural_subset_from_complete_room"
    assert weak["diagnostics"]["objective_proxy_qualified"] is True
    assert weak["diagnostics"]["capture_spacing_factor"] >= 2
    assert weak["diagnostics"]["connectivity_proxy_qualified"] is True
    assert weak["derivation"]["authoritative_sfm_overlap_must_be_revalidated_in_g2b"] is True



def _diag_frame(
    *,
    name: str,
    proxy: float,
    laplacian: float,
    contrast: float,
    edge_density: float,
) -> DiagnosticFrame:
    return DiagnosticFrame(
        name=name,
        width=640,
        height=480,
        texture_proxy=proxy,
        laplacian_variance=laplacian,
        contrast_stddev=contrast,
        edge_density=edge_density,
    )


def test_a1_known_low_texture_positive_control_is_not_rejected_by_log_compression() -> None:
    # Values reproduce the construct-validation regime observed in Test 20K:
    # large raw component reductions but only a ~10% reduction in the old
    # log-compressed aggregate proxy.  The A1 component gate must pass it.
    reference = tuple(
        _diag_frame(
            name=f"ref-{index}",
            proxy=9.368395307,
            laplacian=186.075661789,
            contrast=56.912882076,
            edge_density=0.047849794,
        )
        for index in range(12)
    )
    candidate = tuple(
        _diag_frame(
            name=f"candidate-{index}",
            proxy=8.448143778,
            laplacian=114.717507495,
            contrast=38.112036417,
            edge_density=0.005758464,
        )
        for index in range(12)
    )
    diag = _low_texture_component_diagnostic(
        reference_frames=reference,
        candidate_frames=candidate,
        policy=FreshBindingPolicy(),
        selection="unit_positive_control",
        contiguous_source_order_preserved=True,
    )
    assert diag["legacy_texture_proxy"]["methodology"] == LEGACY_LOW_TEXTURE_METHOD
    assert diag["legacy_texture_proxy"]["median_ratio_to_reference"] > 0.75
    assert diag["legacy_texture_proxy"]["authoritative_for_qualification"] is False
    ratios = diag["component_ratios_to_reference"]
    assert ratios["laplacian_variance"] < 0.75
    assert ratios["contrast_stddev"] < 0.75
    assert ratios["edge_density"] < 0.75
    assert diag["qualification_margin_ratio"] < 0.75
    assert diag["objective_proxy_qualified"] is True


def test_a1_component_gate_requires_edge_reduction_and_a_supporting_component() -> None:
    policy = FreshBindingPolicy()
    reference = tuple(
        _diag_frame(name=f"r-{index}", proxy=10.0, laplacian=100.0, contrast=50.0, edge_density=0.10)
        for index in range(12)
    )
    edge_not_reduced = tuple(
        _diag_frame(name=f"e-{index}", proxy=9.0, laplacian=60.0, contrast=30.0, edge_density=0.09)
        for index in range(12)
    )
    supporting_components_not_reduced = tuple(
        _diag_frame(name=f"s-{index}", proxy=9.0, laplacian=90.0, contrast=45.0, edge_density=0.05)
        for index in range(12)
    )
    first = _low_texture_component_diagnostic(
        reference_frames=reference,
        candidate_frames=edge_not_reduced,
        policy=policy,
        selection="edge_negative_control",
        contiguous_source_order_preserved=True,
    )
    second = _low_texture_component_diagnostic(
        reference_frames=reference,
        candidate_frames=supporting_components_not_reduced,
        policy=policy,
        selection="support_negative_control",
        contiguous_source_order_preserved=True,
    )
    assert first["edge_density_qualified"] is False
    assert first["objective_proxy_qualified"] is False
    assert second["edge_density_qualified"] is True
    assert second["laplacian_variance_qualified"] is False
    assert second["contrast_stddev_qualified"] is False
    assert second["objective_proxy_qualified"] is False


def test_a1_methodology_amendment_is_frozen_and_tamper_evident(tmp_path: Path) -> None:
    snapshot = _ready_snapshot(tmp_path)
    amendment = snapshot["methodology_amendments"]["low_texture"]
    assert amendment["amendment_id"] == LOW_TEXTURE_AMENDMENT_ID
    assert amendment["replacement_method"] == LOW_TEXTURE_METHOD
    assert amendment["predecessor_method"] == LEGACY_LOW_TEXTURE_METHOD
    assert amendment["predecessor_retired_for_qualification"] is True
    assert amendment["provider_quality_results_consumed_before_amendment"] is False
    tampered = json.loads(json.dumps(snapshot))
    tampered["methodology_amendments"]["low_texture"]["replacement_threshold"] = 0.80
    with pytest.raises(ReconstructionValidationError, match="methodology amendment drift"):
        validate_fresh_scenario_binding_snapshot(tampered, verify_files=True)

def test_ready_snapshot_revalidates_pixels_diagnostics_and_review_evidence(tmp_path: Path) -> None:
    snapshot = _ready_snapshot(tmp_path)
    bindings = validate_fresh_scenario_binding_snapshot(snapshot, verify_files=True)
    assert tuple(bindings) == (
        "g2_complete_room_mvs_da3",
        "g2_low_texture_mvs_da3",
        "g2_weak_overlap_mvs_da3",
    )


def test_bound_pixel_mutation_fails_closed(tmp_path: Path) -> None:
    snapshot = _ready_snapshot(tmp_path)
    complete = snapshot["fresh_bindings"][0]
    path = Path(complete["source_identity"]["directory"]) / complete["source_identity"]["files"][0]["name"]
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ReconstructionValidationError, match="changed after binding"):
        validate_fresh_scenario_binding_snapshot(snapshot, verify_files=True)


def test_policy_drift_fails_closed(tmp_path: Path) -> None:
    snapshot = _ready_snapshot(tmp_path)
    snapshot = json.loads(json.dumps(snapshot))
    snapshot["policy"]["weak_overlap_minimum_stride"] = 1
    with pytest.raises(ReconstructionValidationError, match="policy drift"):
        validate_fresh_scenario_binding_snapshot(snapshot, verify_files=True)


def test_tampered_low_texture_subset_fails_recomputation(tmp_path: Path) -> None:
    snapshot = _ready_snapshot(tmp_path)
    snapshot = json.loads(json.dumps(snapshot))
    low = snapshot["fresh_bindings"][1]
    complete = snapshot["fresh_bindings"][0]
    low["source_identity"] = complete["source_identity"]
    with pytest.raises(ReconstructionValidationError, match="low-texture controlled subset drifted"):
        validate_fresh_scenario_binding_snapshot(snapshot, verify_files=True)


def test_inferred_semantic_label_is_rejected(tmp_path: Path) -> None:
    snapshot = _ready_snapshot(tmp_path)
    snapshot = json.loads(json.dumps(snapshot))
    snapshot["fresh_bindings"][0]["semantic_binding_inferred"] = True
    with pytest.raises(ReconstructionValidationError, match="never inferred"):
        validate_fresh_scenario_binding_snapshot(snapshot, verify_files=True)


def test_contact_sheet_mutation_fails_closed(tmp_path: Path) -> None:
    snapshot = _ready_snapshot(tmp_path)
    review = snapshot["fresh_bindings"][0]["visual_review"]
    path = Path(review["path"])
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ReconstructionValidationError, match="visual-review evidence changed"):
        validate_fresh_scenario_binding_snapshot(snapshot, verify_files=True)
