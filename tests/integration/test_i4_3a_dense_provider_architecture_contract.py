from __future__ import annotations

from pathlib import Path

from core.reconstruction.dense_provider_architecture import (
    DenseProviderFamily,
    DenseProviderImplementation,
    DenseProviderProgressStage,
    DenseProviderSelectionMode,
    build_i4_3a_architecture_snapshot,
    dense_provider_implementation_contracts,
)

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core/reconstruction/dense_provider_architecture.py"
RUNNER = ROOT / "scripts/run_i4_3a_architecture_smoke.py"
TEST_RUNNER = ROOT / "scripts/run_i4_3a_tests.sh"
DOC = ROOT / "docs/phase_history/I4_3A_CANONICAL_DENSE_PROVIDER_ARCHITECTURE_FOUNDATION.md"


def _closure() -> dict:
    return {
        "phase": "I4.2F",
        "status": "pass",
        "pass": True,
        "decision": {
            "i4_2_closed": True,
            "i4_2f_closed": True,
            "i4_3_provider_architecture_authorized": True,
            "dense_provider_promoted": False,
            "production_d12_policy_change_authorized": False,
        },
        "authority": {
            "authoritative_sparse_modified": False,
            "bundle_adjustment_rerun": False,
            "certified_metrology": False,
            "dense_provider_promoted": False,
            "measurement_authoritative": False,
            "metric_scale_claimed": False,
            "patch_match_rerun": False,
            "pose_refinement": False,
            "production_filter_policy_changed": False,
            "production_manifold_applied_to_authoritative_scene": False,
            "sfm_rerun": False,
            "stereo_fusion_rerun": False,
            "surface_mesh_generated": False,
        },
        "scientific_conclusion": {
            "d12_sparse_safe_on_temple_ring_within_tested_transfer": True,
            "global_universal_detail_safety_claimed": False,
            "temple_ring_detail_preservation_qualified": True,
            "temple_ring_driven_d12_policy_correction_required": False,
        },
    }


def test_i43a_files_are_present_and_phase_is_documented() -> None:
    assert CORE.is_file()
    assert RUNNER.is_file()
    assert TEST_RUNNER.is_file()
    assert DOC.is_file()
    assert "I4.3A" in DOC.read_text(encoding="utf-8")


def test_i43a_foundation_separates_selection_family_and_implementation() -> None:
    assert [mode.value for mode in DenseProviderSelectionMode] == ["auto", "explicit_provider"]
    assert [family.value for family in DenseProviderFamily] == [
        "mvs",
        "multi_view_ai_depth",
        "temporal_depth",
        "single_image_depth",
        "hybrid",
    ]
    assert len(DenseProviderImplementation) == 10


def test_i43a_implementation_catalog_contains_all_locked_provider_targets() -> None:
    contracts = dense_provider_implementation_contracts()
    assert {item.implementation for item in contracts} == set(DenseProviderImplementation)
    assert {item.integration_stage for item in contracts} == {"I4.3B", "I4.3C", "I4.3D", "I4.3E"}


def test_i43a_snapshot_locks_progressive_preview_without_ui_binding() -> None:
    snapshot = build_i4_3a_architecture_snapshot(i4_2f_closure=_closure())
    progress = snapshot["progressive_preview_contract"]
    assert progress["required_for_v1_0"] is True
    assert progress["ui_implemented_in_i4_3a"] is False
    assert progress["progress_stages"] == [stage.value for stage in DenseProviderProgressStage]
    assert progress["provisional_geometry_authoritative"] is False
    assert progress["final_canonical_geometry_remains_separate"] is True


def test_i43a_snapshot_locks_full_room_mvs_vs_da3_and_no_universal_winner() -> None:
    snapshot = build_i4_3a_architecture_snapshot(i4_2f_closure=_closure())
    assert snapshot["qualification_plan"]["complete_room_mvs_vs_da3_required"] is True
    assert snapshot["architecture"]["scenario_specific_qualification_required"] is True
    assert snapshot["architecture"]["universal_provider_winner_assumed"] is False


def test_i43a_snapshot_keeps_hybrid_as_unapproved_hypothesis() -> None:
    snapshot = build_i4_3a_architecture_snapshot(i4_2f_closure=_closure())
    assert snapshot["qualification_plan"]["hybrid_family_defined_as_future_hypothesis"] is True
    assert snapshot["qualification_plan"]["hybrid_implementation_authorized"] is False
    assert snapshot["authority"]["hybrid_implementation_authorized"] is False


def test_i43a_foundation_does_not_execute_heavy_backends_or_change_policy() -> None:
    text = (CORE.read_text(encoding="utf-8") + RUNNER.read_text(encoding="utf-8")).lower()
    forbidden_calls = (
        "patch_match_stereo(",
        "stereo_fusion(",
        "undistort_images(",
        "incremental_mapping(",
        "bundle_adjustment(",
        "subprocess.run(",
        "depthanything3providerexecutor(",
        "run_temple_ring_mvs(",
    )
    for token in forbidden_calls:
        assert token not in text
    assert '"provider_promoted": false' in text
    assert '"production_filter_policy_changed": false' in text
    assert '"backend_started": false' in text


def test_i43a_does_not_modify_existing_provider_or_ui_modules() -> None:
    doc = DOC.read_text(encoding="utf-8")
    for path in (
        "core/reconstruction/mvs_execution.py",
        "core/reconstruction/da3_provider.py",
        "core/models/adapters/video_depth_anything_adapter.py",
        "app/reconstruction_controller.py",
        "app/reconstruction_dense_provider_presentation.py",
    ):
        assert path in doc
    runner_text = TEST_RUNNER.read_text(encoding="utf-8")
    assert "test_i4_2f_temple_ring_closure.py" in runner_text
    assert "test_i4_3a_dense_provider_architecture.py" in runner_text


def test_i43a_next_steps_cover_i43b_through_i43h() -> None:
    snapshot = build_i4_3a_architecture_snapshot(i4_2f_closure=_closure())
    roadmap = snapshot["i4_3_roadmap"]
    assert [entry.split()[0] for entry in roadmap] == [
        "I4.3B",
        "I4.3C",
        "I4.3D",
        "I4.3E",
        "I4.3F",
        "I4.3G",
        "I4.3H",
    ]
    assert snapshot["next"].startswith("I4.3B MVS production provider adapter")
    assert snapshot["authority"]["provider_selected"] is False
