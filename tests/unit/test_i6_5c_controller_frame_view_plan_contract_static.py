from pathlib import Path


def test_i6_5c_controller_surface_is_intent_only() -> None:
    text = Path("app/reconstruction_controller.py").read_text(encoding="utf-8")
    assert "reconstruction_frame_view_plan_changed = pyqtSignal(object)" in text
    assert "def reconstruction_frame_view_plan_state(" in text
    assert "def set_reconstruction_frame_view_resource_snapshot(" in text
    assert "def set_reconstruction_frame_view_plan(" in text
    assert "def reset_reconstruction_frame_view_plan(" in text

    start = text.index("def set_reconstruction_frame_view_plan(")
    end = text.index("def reset_reconstruction_frame_view_plan(", start)
    block = text[start:end]
    for forbidden in (
        "start_offline",
        "start_selected_offline",
        "_offline_controller.start",
        "select_dense_provider",
        "build_offline_reconstruction_command",
        "i5_auto_policy_engine",
    ):
        assert forbidden not in block


def test_i6_5c_state_module_is_dependency_light_and_execution_free() -> None:
    text = Path("app/reconstruction_frame_view_plan_state.py").read_text(encoding="utf-8")
    for forbidden in (
        "PyQt6",
        "subprocess",
        "import torch",
        "import pycolmap",
        "start_offline",
        "select_dense_provider",
        "i5_auto_policy_engine",
    ):
        assert forbidden not in text
    for required in (
        "FrameViewPlanStateStatus",
        "FrameViewPlanIntentOrigin",
        "ReconstructionFrameViewPlanState",
        "source_provenance_from_selection",
        "build_reconstruction_frame_view_plan_state",
        "selection_metadata_manifest_v1",
    ):
        assert required in text


def test_i6_5c_worker_and_ui_are_not_referenced_by_new_state() -> None:
    text = Path("app/reconstruction_frame_view_plan_state.py").read_text(encoding="utf-8")
    assert "offline_reconstruction_worker" not in text
    assert "reconstruction_page" not in text
