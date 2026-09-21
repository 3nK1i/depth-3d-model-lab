from __future__ import annotations

from core.reconstruction.dense_provider_architecture import DenseProviderFamily, DenseProviderImplementation
from core.reconstruction.reconstruction_provider_selection_contract import (
    ProductionProviderAvailability,
    ProductionProviderCatalogEntry,
    ProductionProviderQualification,
    ProductionProviderSelectionMode,
    ProductionProviderSelectionSnapshot,
    canonical_production_provider_catalog,
)


def _qualified(provider_id: str, implementation: DenseProviderImplementation, label: str, digest: str) -> ProductionProviderCatalogEntry:
    family = {
        DenseProviderImplementation.COLMAP_MVS: DenseProviderFamily.MVS,
        DenseProviderImplementation.DA3: DenseProviderFamily.MULTI_VIEW_AI_DEPTH,
        DenseProviderImplementation.ROLLING_DEPTH: DenseProviderFamily.TEMPORAL_DEPTH,
        DenseProviderImplementation.VIDEO_DEPTH_ANYTHING: DenseProviderFamily.TEMPORAL_DEPTH,
        DenseProviderImplementation.METRIC3D_V2: DenseProviderFamily.SINGLE_IMAGE_DEPTH,
        DenseProviderImplementation.UNIDEPTH_V2: DenseProviderFamily.SINGLE_IMAGE_DEPTH,
        DenseProviderImplementation.MOGE_2: DenseProviderFamily.SINGLE_IMAGE_DEPTH,
        DenseProviderImplementation.DEPTH_PRO: DenseProviderFamily.SINGLE_IMAGE_DEPTH,
        DenseProviderImplementation.DEPTH_ANYTHING_V2: DenseProviderFamily.SINGLE_IMAGE_DEPTH,
        DenseProviderImplementation.MARIGOLD: DenseProviderFamily.SINGLE_IMAGE_DEPTH,
    }[implementation]
    return ProductionProviderCatalogEntry(
        provider_id=provider_id,
        label=label,
        implementation=implementation,
        family=family,
        availability=ProductionProviderAvailability.AVAILABLE,
        qualification=ProductionProviderQualification.QUALIFIED_FOR_CONTEXT,
        qualification_context_id="stress-context",
        qualification_evidence_sha256=(digest * 64,),
        manual_selection_allowed=True,
    )


def test_i6_4a_canonical_catalog_is_deterministic_over_repeated_construction() -> None:
    baseline = [entry.as_dict() for entry in canonical_production_provider_catalog()]
    for _ in range(100):
        assert [entry.as_dict() for entry in canonical_production_provider_catalog()] == baseline


def test_i6_4a_recommendation_and_selection_remain_independent_over_repetition() -> None:
    depth_pro = _qualified(
        "depth-pro-single-image",
        DenseProviderImplementation.DEPTH_PRO,
        "Depth Pro",
        "a",
    )
    metric = _qualified(
        "metric3d-v2-single-image",
        DenseProviderImplementation.METRIC3D_V2,
        "Metric3D v2",
        "b",
    )
    for _ in range(100):
        snapshot = ProductionProviderSelectionSnapshot(
            providers=(depth_pro, metric),
            recommended_provider_id=depth_pro.provider_id,
            selection_mode=ProductionProviderSelectionMode.MANUAL,
            selected_provider_id=metric.provider_id,
        )
        assert snapshot.recommended_provider_id == depth_pro.provider_id
        assert snapshot.selected_provider_id == metric.provider_id
        assert snapshot.provider_execution_authorized is False


def test_i6_4a_default_catalog_never_exposes_auto_as_provider() -> None:
    for _ in range(100):
        assert "auto" not in {entry.provider_id for entry in canonical_production_provider_catalog()}


def test_i6_4a_auto_gate_remains_closed_in_repeated_snapshots() -> None:
    catalog = canonical_production_provider_catalog()
    for _ in range(100):
        snapshot = ProductionProviderSelectionSnapshot(providers=catalog)
        assert snapshot.auto_gate.offline_shadow_qualified is True
        assert snapshot.auto_gate.production_qualified is False
        assert snapshot.auto_gate.production_authorized is False
        assert snapshot.auto_gate.selectable is False
