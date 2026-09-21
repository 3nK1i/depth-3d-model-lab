from __future__ import annotations

from core.reconstruction.reconstruction_production_auto_gate import (
    build_production_auto_qualification_boundary,
)
from core.reconstruction.reconstruction_provider_selection_contract import (
    ProductionProviderSelectionSnapshot,
    canonical_production_provider_catalog,
)


def _snapshot() -> ProductionProviderSelectionSnapshot:
    return ProductionProviderSelectionSnapshot(
        providers=canonical_production_provider_catalog(),
    )


def test_i6_4c_repeated_gate_composition_is_deterministic() -> None:
    snapshot = _snapshot()
    first = build_production_auto_qualification_boundary(snapshot)
    for _ in range(500):
        assert build_production_auto_qualification_boundary(snapshot) == first


def test_i6_4c_repeated_gate_never_becomes_selectable() -> None:
    snapshot = _snapshot()
    for _ in range(500):
        boundary = build_production_auto_qualification_boundary(snapshot)
        assert boundary.production_auto_qualified is False
        assert boundary.production_auto_authorized is False
        assert boundary.selectable is False
        assert boundary.provider_execution_authorized is False
