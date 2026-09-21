from pathlib import Path


PLANNER = Path("core/reconstruction/reconstruction_frame_view_planning.py")


def test_i6_5b_planner_stays_core_only_and_dependency_light():
    text = PLANNER.read_text(encoding="utf-8")
    for forbidden in (
        "from app.",
        "import app.",
        "PyQt6",
        "subprocess",
        "import torch",
        "import pycolmap",
        "i5_auto_policy_engine",
    ):
        assert forbidden not in text


def test_i6_5b_reuses_existing_i4_3f2_resource_planner_instead_of_duplicating_formula_logic():
    text = PLANNER.read_text(encoding="utf-8")
    assert "build_adaptive_pre_sfm_frame_plan" in text
    assert "legacy_i4_3f2_resource_planner_reused" in text
    assert "I43F2_PRE_SFM_WORK_UNITS_PER_FRAME" in text


def test_i6_5b_has_no_sampling_execution_or_retry_entrypoint():
    text = PLANNER.read_text(encoding="utf-8")
    for forbidden in (
        "def start_",
        "def execute_",
        "def retry_",
        "while not pass",
        "retry_until_pass=True",
        "select_dense_provider",
    ):
        assert forbidden not in text
    assert '"retry_until_pass": False' in text
    assert '"silent_replan_after_sfm_failure": False' in text


def test_i6_5b_preserves_pycolmap_authority_and_no_global_cap():
    text = PLANNER.read_text(encoding="utf-8")
    assert "I65A_CAMERA_AUTHORITY_OWNER" in text
    assert '"planned_views_are_registered_cameras": False' in text
    assert "I65B_PERMANENT_GLOBAL_VIEW_CAP = None" in text
