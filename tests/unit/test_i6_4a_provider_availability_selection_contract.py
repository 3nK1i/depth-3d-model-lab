from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from core.reconstruction.dense_provider_architecture import (
    DenseProviderFamily,
    DenseProviderImplementation,
)
from core.reconstruction.reconstruction_provider_selection_contract import (
    I64A_CANONICAL_PROVIDER_COUNT,
    I64A_PRODUCTION_AUTO_POLICY_ID,
    ProductionAutoGate,
    ProductionProviderAvailability,
    ProductionProviderCatalogEntry,
    ProductionProviderQualification,
    ProductionProviderSelectionMode,
    ProductionProviderSelectionSnapshot,
    canonical_production_provider_catalog,
)


def _entry(
    *,
    provider_id: str = "depth-pro-single-image",
    implementation: DenseProviderImplementation = DenseProviderImplementation.DEPTH_PRO,
    family: DenseProviderFamily = DenseProviderFamily.SINGLE_IMAGE_DEPTH,
    availability: ProductionProviderAvailability = ProductionProviderAvailability.AVAILABLE,
    qualification: ProductionProviderQualification = ProductionProviderQualification.QUALIFIED_FOR_CONTEXT,
    selectable: bool = True,
) -> ProductionProviderCatalogEntry:
    return ProductionProviderCatalogEntry(
        provider_id=provider_id,
        label="Depth Pro",
        implementation=implementation,
        family=family,
        availability=availability,
        availability_reason_codes=() if availability is ProductionProviderAvailability.AVAILABLE else ("runtime_unavailable",),
        qualification=qualification,
        qualification_context_id=("ctx-qualified" if qualification is not ProductionProviderQualification.UNRESOLVED else ""),
        qualification_evidence_sha256=("a" * 64,) if qualification is not ProductionProviderQualification.UNRESOLVED else (),
        manual_selection_allowed=selectable,
    )


def test_i6_4a_canonical_catalog_has_all_ten_concrete_providers_and_no_auto() -> None:
    catalog = canonical_production_provider_catalog()
    assert len(catalog) == I64A_CANONICAL_PROVIDER_COUNT == 10
    assert len({entry.provider_id for entry in catalog}) == 10
    assert {entry.implementation for entry in catalog} == set(DenseProviderImplementation)
    assert all(entry.provider_id != I64A_PRODUCTION_AUTO_POLICY_ID for entry in catalog)
    assert all(entry.availability is ProductionProviderAvailability.UNKNOWN for entry in catalog)
    assert all(entry.qualification is ProductionProviderQualification.UNRESOLVED for entry in catalog)
    assert all(entry.manual_selection_allowed is False for entry in catalog)


def test_i6_4a_catalog_entry_is_immutable() -> None:
    entry = _entry()
    with pytest.raises(FrozenInstanceError):
        entry.manual_selection_allowed = False  # type: ignore[misc]


def test_i6_4a_available_provider_cannot_carry_unavailability_reason() -> None:
    with pytest.raises(ValueError, match="available provider"):
        ProductionProviderCatalogEntry(
            provider_id="depth-pro-single-image",
            label="Depth Pro",
            implementation=DenseProviderImplementation.DEPTH_PRO,
            family=DenseProviderFamily.SINGLE_IMAGE_DEPTH,
            availability=ProductionProviderAvailability.AVAILABLE,
            availability_reason_codes=("should_not_exist",),
            qualification=ProductionProviderQualification.UNRESOLVED,
            manual_selection_allowed=False,
        )


def test_i6_4a_unavailable_provider_requires_reason() -> None:
    with pytest.raises(ValueError, match="unavailable provider"):
        ProductionProviderCatalogEntry(
            provider_id="depth-pro-single-image",
            label="Depth Pro",
            implementation=DenseProviderImplementation.DEPTH_PRO,
            family=DenseProviderFamily.SINGLE_IMAGE_DEPTH,
            availability=ProductionProviderAvailability.UNAVAILABLE,
            availability_reason_codes=(),
            qualification=ProductionProviderQualification.UNRESOLVED,
            manual_selection_allowed=False,
        )


def test_i6_4a_manual_selection_requires_available_and_context_qualified() -> None:
    for availability, qualification in (
        (ProductionProviderAvailability.UNKNOWN, ProductionProviderQualification.QUALIFIED_FOR_CONTEXT),
        (ProductionProviderAvailability.UNAVAILABLE, ProductionProviderQualification.QUALIFIED_FOR_CONTEXT),
        (ProductionProviderAvailability.AVAILABLE, ProductionProviderQualification.UNRESOLVED),
        (ProductionProviderAvailability.AVAILABLE, ProductionProviderQualification.NOT_QUALIFIED_FOR_CONTEXT),
    ):
        with pytest.raises(ValueError, match="manual selection"):
            _entry(availability=availability, qualification=qualification, selectable=True)


def test_i6_4a_qualified_state_requires_context_and_sha256_provenance() -> None:
    with pytest.raises(ValueError, match="qualification_context_id"):
        ProductionProviderCatalogEntry(
            provider_id="depth-pro-single-image",
            label="Depth Pro",
            implementation=DenseProviderImplementation.DEPTH_PRO,
            family=DenseProviderFamily.SINGLE_IMAGE_DEPTH,
            availability=ProductionProviderAvailability.AVAILABLE,
            qualification=ProductionProviderQualification.QUALIFIED_FOR_CONTEXT,
            qualification_context_id="",
            qualification_evidence_sha256=("a" * 64,),
            manual_selection_allowed=False,
        )
    with pytest.raises(ValueError, match="SHA-256"):
        ProductionProviderCatalogEntry(
            provider_id="depth-pro-single-image",
            label="Depth Pro",
            implementation=DenseProviderImplementation.DEPTH_PRO,
            family=DenseProviderFamily.SINGLE_IMAGE_DEPTH,
            availability=ProductionProviderAvailability.AVAILABLE,
            qualification=ProductionProviderQualification.QUALIFIED_FOR_CONTEXT,
            qualification_context_id="ctx",
            qualification_evidence_sha256=("bad",),
            manual_selection_allowed=False,
        )


def test_i6_4a_unresolved_qualification_cannot_carry_evidence_claims() -> None:
    with pytest.raises(ValueError, match="unresolved qualification"):
        ProductionProviderCatalogEntry(
            provider_id="depth-pro-single-image",
            label="Depth Pro",
            implementation=DenseProviderImplementation.DEPTH_PRO,
            family=DenseProviderFamily.SINGLE_IMAGE_DEPTH,
            availability=ProductionProviderAvailability.AVAILABLE,
            qualification=ProductionProviderQualification.UNRESOLVED,
            qualification_context_id="ctx",
            qualification_evidence_sha256=("a" * 64,),
            manual_selection_allowed=False,
        )


def test_i6_4a_provider_family_must_match_canonical_implementation() -> None:
    with pytest.raises(ValueError, match="canonical implementation"):
        _entry(family=DenseProviderFamily.MVS)


def test_i6_4a_auto_gate_is_fail_closed_and_separate_from_provider_catalog() -> None:
    gate = ProductionAutoGate()
    assert gate.policy_id == "auto"
    assert gate.offline_shadow_qualified is True
    assert gate.production_qualified is False
    assert gate.production_authorized is False
    assert gate.selectable is False
    assert gate.reason_codes == ("production_auto_not_qualified", "production_auto_not_authorized")

    with pytest.raises(ValueError, match="remain fail-closed"):
        ProductionAutoGate(production_qualified=True)
    with pytest.raises(ValueError, match="remain fail-closed"):
        ProductionAutoGate(production_authorized=True)
    with pytest.raises(ValueError, match="remain fail-closed"):
        ProductionAutoGate(selectable=True)


def test_i6_4a_snapshot_distinguishes_recommended_selected_and_execution() -> None:
    recommended = _entry(provider_id="depth-pro-single-image")
    selected = ProductionProviderCatalogEntry(
        provider_id="metric3d-v2-single-image",
        label="Metric3D v2",
        implementation=DenseProviderImplementation.METRIC3D_V2,
        family=DenseProviderFamily.SINGLE_IMAGE_DEPTH,
        availability=ProductionProviderAvailability.AVAILABLE,
        qualification=ProductionProviderQualification.QUALIFIED_FOR_CONTEXT,
        qualification_context_id="ctx-qualified",
        qualification_evidence_sha256=("b" * 64,),
        manual_selection_allowed=True,
    )
    snapshot = ProductionProviderSelectionSnapshot(
        providers=(recommended, selected),
        recommended_provider_id=recommended.provider_id,
        selection_mode=ProductionProviderSelectionMode.MANUAL,
        selected_provider_id=selected.provider_id,
    )
    assert snapshot.recommended_provider_id != snapshot.selected_provider_id
    assert snapshot.provider_selection_effect is False
    assert snapshot.provider_activation_authorized is False
    assert snapshot.provider_execution_authorized is False
    assert snapshot.start_action_authorized is False


def test_i6_4a_recommendation_does_not_imply_selection() -> None:
    entry = _entry()
    snapshot = ProductionProviderSelectionSnapshot(
        providers=(entry,),
        recommended_provider_id=entry.provider_id,
    )
    assert snapshot.selection_mode is ProductionProviderSelectionMode.NONE
    assert snapshot.selected_provider_id == ""


def test_i6_4a_manual_selection_must_reference_selectable_catalog_provider() -> None:
    entry = _entry(selectable=False)
    with pytest.raises(ValueError, match="not manually selectable"):
        ProductionProviderSelectionSnapshot(
            providers=(entry,),
            selection_mode=ProductionProviderSelectionMode.MANUAL,
            selected_provider_id=entry.provider_id,
        )


def test_i6_4a_auto_selection_mode_is_not_constructible_while_gate_is_closed() -> None:
    with pytest.raises(ValueError, match="Auto selection"):
        ProductionProviderSelectionSnapshot(
            providers=canonical_production_provider_catalog(),
            selection_mode=ProductionProviderSelectionMode.AUTO,
        )


def test_i6_4a_serialization_keeps_claim_boundaries_explicit() -> None:
    entry = _entry()
    payload = ProductionProviderSelectionSnapshot(providers=(entry,)).as_dict()
    assert payload["effects"]["provider_selection_effect"] is False
    assert payload["effects"]["provider_activation_authorized"] is False
    assert payload["effects"]["provider_execution_authorized"] is False
    assert payload["effects"]["start_action_authorized"] is False
    assert payload["auto"]["production_qualified"] is False
    assert payload["auto"]["production_authorized"] is False
    assert payload["claim_boundaries"]["global_provider_ranking"] is False
    assert payload["claim_boundaries"]["global_scalar_score"] is False
    assert payload["authority"]["camera_authority_owner"] == "pycolmap_sfm"
    assert payload["authority"]["certified_metrology"] is False
