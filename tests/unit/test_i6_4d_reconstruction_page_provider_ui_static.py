from pathlib import Path

PAGE = Path("app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
PRESENTATION = Path("app/reconstruction_provider_selection_presentation.py").read_text(encoding="utf-8")


def test_i6_4d_page_contains_distinct_provider_auto_group() -> None:
    assert 'self._settings_group("PROVIDER & AUTO")' in PAGE
    assert 'setObjectName("reconstructionProviderSelectionCombo")' in PAGE
    assert 'QPushButton("Auto unavailable")' in PAGE
    assert "self._provider_auto_button.setEnabled(False)" in PAGE


def test_i6_4d_page_binds_passive_selection_signal_and_initial_state() -> None:
    assert "controller.reconstruction_provider_selection_changed.connect(" in PAGE
    assert "self.set_reconstruction_provider_selection_state" in PAGE
    assert "controller.reconstruction_provider_selection_state" in PAGE


def test_i6_4d_manual_handler_records_intent_only() -> None:
    start = PAGE.index("def _on_reconstruction_provider_activated(")
    end = PAGE.index("def _on_reconstruction_provider_clear_clicked(", start)
    block = PAGE[start:end]
    assert "controller.select_reconstruction_provider(provider_id)" in block
    for forbidden in (
        "start_offline",
        "start_selected_offline",
        "_offline_controller.start",
        "select_dense_provider",
        "activate_provider(",
        "i5_auto_policy_engine",
    ):
        assert forbidden not in block


def test_i6_4d_auto_has_no_click_handler_or_activation_surface() -> None:
    assert "_provider_auto_button.clicked.connect" not in PAGE
    for forbidden in (
        "select_reconstruction_auto",
        "activate_reconstruction_auto",
        "start_reconstruction_auto",
        "evaluate_i5_13b_auto_policy",
    ):
        assert forbidden not in PAGE


def test_i6_4d_presentation_module_is_execution_free() -> None:
    for forbidden in (
        "PyQt6",
        "subprocess",
        "import torch",
        "import pycolmap",
        "i5_auto_policy_engine",
        "start_offline",
        "activate_provider",
    ):
        assert forbidden not in PRESENTATION


def test_i6_4d_preserves_i6_3d_primary_stack_before_new_group() -> None:
    start = PAGE.index("primary_groups = (")
    end = PAGE.index(")", start)
    block = PAGE[start:end]
    order = [
        "self._digital_twin_workspace_group",
        "self._digital_twin_processing_group",
        "self._canonical_scene_group",
        "self._estimated_ruler_scale_group",
        "self._reconstruction_advisor_group",
        "self._dense_rgb_group",
        "self._reconstruction_provider_group",
    ]
    positions = [block.index(item) for item in order]
    assert positions == sorted(positions)
