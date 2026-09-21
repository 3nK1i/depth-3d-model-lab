from __future__ import annotations

from app.reconstruction_provider_selection_presentation import (
    present_reconstruction_provider_auto,
)
from app.reconstruction_provider_selection_state import (
    ManualProviderSelectionStatus,
    build_reconstruction_provider_selection_state,
)
from core.reconstruction.reconstruction_production_auto_gate import (
    build_production_auto_qualification_boundary,
)
from tests.i6_4d_helpers import advisor_state


def _presentation(*, selected: str = "", recommend: str = "", source_ready: bool = True):
    state = build_reconstruction_provider_selection_state(
        advisor_state=advisor_state(source_ready=source_ready, recommend=recommend),
        selected_provider_id=selected,
    )
    gate = build_production_auto_qualification_boundary(state.snapshot)
    return state, present_reconstruction_provider_auto(state, gate)


def test_i6_4d_no_source_is_neutral_fail_closed() -> None:
    state, ui = _presentation(source_ready=False)
    assert state.status is ManualProviderSelectionStatus.NONE
    assert ui.status_text == "Waiting for source"
    assert ui.status_kind == "neutral"
    assert ui.combo_enabled is False
    assert ui.clear_enabled is False
    assert ui.auto_text == "Not qualified"
    assert ui.auto_status_kind == "warning"


def test_i6_4d_exact_available_qualified_provider_is_selectable() -> None:
    _state, ui = _presentation()
    selectable = [option for option in ui.options if option.selectable]
    assert ui.status_text == "Ready for manual choice"
    assert ui.status_kind == "ready"
    assert ui.combo_enabled is True
    assert ui.eligibility_text == "1 / 10 selectable"
    assert [option.provider_id for option in selectable] == ["colmap-mvs-external"]


def test_i6_4d_recommendation_is_marked_but_not_selected() -> None:
    state, ui = _presentation(recommend="colmap-mvs-external")
    assert state.selected_provider_id == ""
    assert ui.recommendation_text == "COLMAP MVS"
    option = next(item for item in ui.options if item.provider_id == "colmap-mvs-external")
    assert option.recommended is True
    assert option.selected is False
    assert option.label.endswith("— Recommended")


def test_i6_4d_manual_selection_is_presented_as_intent_only() -> None:
    state, ui = _presentation(selected="colmap-mvs-external")
    assert state.manual_selection_active is True
    assert ui.status_text == "Manual selection recorded"
    assert ui.status_kind == "ready"
    assert ui.selection_text == "COLMAP MVS"
    assert ui.clear_enabled is True
    assert ui.boundary_text == "Intent only • no provider activated or started"


def test_i6_4d_invalidated_selection_is_warning_not_runtime_error() -> None:
    state = build_reconstruction_provider_selection_state(
        advisor_state=advisor_state(source_ready=False),
        selected_provider_id="colmap-mvs-external",
    )
    gate = build_production_auto_qualification_boundary(state.snapshot)
    ui = present_reconstruction_provider_auto(state, gate)
    assert state.status is ManualProviderSelectionStatus.CLEARED_INVALID
    assert ui.status_text == "Selection cleared"
    assert ui.status_kind == "warning"
    assert "ready offline source" in ui.selection_tooltip


def test_i6_4d_auto_remains_visible_and_fail_closed() -> None:
    state, ui = _presentation(selected="colmap-mvs-external")
    gate = build_production_auto_qualification_boundary(state.snapshot)
    assert gate.offline_shadow_qualified is True
    assert gate.production_auto_qualified is False
    assert gate.production_auto_authorized is False
    assert gate.selectable is False
    assert ui.auto_text == "Not qualified"
    assert "Offline/shadow Auto is qualified" in ui.auto_tooltip


def test_i6_4d_all_canonical_providers_remain_visible_in_eligibility_metadata() -> None:
    _state, ui = _presentation()
    assert len(ui.options) == 10
    assert "COLMAP MVS" in ui.eligibility_tooltip
    assert "Depth Pro" in ui.eligibility_tooltip
    assert "Marigold" in ui.eligibility_tooltip


def test_i6_4d_presentation_has_no_execution_authority() -> None:
    _state, ui = _presentation()
    assert "no provider activated or started" in ui.boundary_text
    assert "No provider activation" in ui.boundary_tooltip
