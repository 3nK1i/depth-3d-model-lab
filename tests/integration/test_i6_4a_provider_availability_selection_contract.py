from __future__ import annotations

from core.reconstruction.dense_provider_architecture import DenseProviderImplementation
from core.reconstruction.i5_auto_policy_engine import I513B_PROVIDER_ID_BY_IMPLEMENTATION
from core.reconstruction.reconstruction_provider_selection_contract import (
    PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION,
    ProductionProviderSelectionSnapshot,
    canonical_production_provider_catalog,
)


def test_i6_4a_provider_ids_match_frozen_i5_canonical_identities() -> None:
    assert PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION == I513B_PROVIDER_ID_BY_IMPLEMENTATION
    assert set(PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION) == set(DenseProviderImplementation)


def test_i6_4a_default_snapshot_is_execution_free_and_auto_fail_closed() -> None:
    snapshot = ProductionProviderSelectionSnapshot(
        providers=canonical_production_provider_catalog(),
    )
    payload = snapshot.as_dict()
    assert len(payload["providers"]) == 10
    assert payload["selected_provider_id"] == ""
    assert payload["recommended_provider_id"] == ""
    assert payload["selection_mode"] == "none"
    assert payload["auto"]["selectable"] is False
    assert payload["effects"] == {
        "provider_selection_effect": False,
        "provider_activation_authorized": False,
        "provider_execution_authorized": False,
        "settings_mutation_authorized": False,
        "start_action_authorized": False,
    }


def test_i6_4a_catalog_order_is_stable_and_tracks_architecture_enum_order() -> None:
    catalog = canonical_production_provider_catalog()
    assert tuple(entry.implementation for entry in catalog) == tuple(DenseProviderImplementation)
