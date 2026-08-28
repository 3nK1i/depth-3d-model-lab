from __future__ import annotations

from pathlib import Path
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_da3_production_comparison import (
    I43G2B_FRESH_SCENARIOS,
    build_g2b_runtime_preflight_snapshot,
    resolve_g2b_scenario_binding,
    validate_g2b_predecessors,
    validate_provider_run_evidence,
)
from tests.i4_3g2b_helpers import da3_runtime, g1_snapshot, g2a_snapshot, make_source, mvs_runtime


def test_predecessors_and_complete_room_binding_pass(tmp_path: Path) -> None:
    source = make_source(tmp_path / "images")
    g1 = g1_snapshot()
    g2a = g2a_snapshot(source)
    validate_g2b_predecessors(i4_3g1_snapshot=g1, i4_3g2a_snapshot=g2a)
    binding = resolve_g2b_scenario_binding(
        i4_3g1_snapshot=g1,
        i4_3g2a_snapshot=g2a,
        scenario_id="g2_complete_room_mvs_da3",
        verify_files=True,
    )
    assert binding.source_image_count == 6
    assert binding.matching_strategy == "sequential"
    assert binding.scenario_id in I43G2B_FRESH_SCENARIOS


def test_house_new_execution_is_fail_closed(tmp_path: Path) -> None:
    source = make_source(tmp_path / "images")
    with pytest.raises(ReconstructionValidationError, match="historical_read_only"):
        resolve_g2b_scenario_binding(
            i4_3g1_snapshot=g1_snapshot(),
            i4_3g2a_snapshot=g2a_snapshot(source),
            scenario_id="g2_house_mvs_da3_historical",
        )


def test_source_hash_drift_is_rejected(tmp_path: Path) -> None:
    source = make_source(tmp_path / "images")
    g2a = g2a_snapshot(source)
    (source / "frame_000.png").write_bytes(b"tampered")
    with pytest.raises(ReconstructionValidationError, match="source identity changed"):
        resolve_g2b_scenario_binding(
            i4_3g1_snapshot=g1_snapshot(),
            i4_3g2a_snapshot=g2a,
            scenario_id="g2_complete_room_mvs_da3",
        )


def test_runtime_preflight_authorizes_only_both_qualified_runtimes(tmp_path: Path) -> None:
    source = make_source(tmp_path / "images")
    kwargs = dict(
        i4_3g1_snapshot=g1_snapshot(),
        i4_3g2a_snapshot=g2a_snapshot(source),
        scenario_id="g2_complete_room_mvs_da3",
        verify_files=True,
    )
    ready = build_g2b_runtime_preflight_snapshot(**kwargs, mvs_runtime=mvs_runtime(), da3_runtime=da3_runtime())
    assert ready["pass"] is True
    assert ready["decision"]["production_native_real_execution_authorized"] is True
    assert ready["scope"]["authoritative_sfm_started"] is False
    assert ready["scope"]["mvs_started"] is False
    assert ready["scope"]["da3_started"] is False
    blocked = build_g2b_runtime_preflight_snapshot(**kwargs, mvs_runtime=mvs_runtime(), da3_runtime=da3_runtime(ready=False))
    assert blocked["pass"] is False
    assert blocked["decision"]["production_native_real_execution_authorized"] is False


def test_mvs_runtime_requires_qualified_cuda_distribution_and_version(tmp_path: Path) -> None:
    source = make_source(tmp_path / "images")
    runtime = mvs_runtime()
    runtime["cuda_distribution"] = False
    snap = build_g2b_runtime_preflight_snapshot(
        i4_3g1_snapshot=g1_snapshot(), i4_3g2a_snapshot=g2a_snapshot(source),
        scenario_id="g2_complete_room_mvs_da3", mvs_runtime=runtime, da3_runtime=da3_runtime(), verify_files=True,
    )
    assert snap["runtime"]["mvs_ready"] is False
    assert snap["pass"] is False


def test_authority_drift_is_rejected(tmp_path: Path) -> None:
    source = make_source(tmp_path / "images")
    g2a = g2a_snapshot(source)
    g2a["authority"]["metric_scale_claimed"] = True
    with pytest.raises(ReconstructionValidationError, match="authority drift"):
        validate_g2b_predecessors(i4_3g1_snapshot=g1_snapshot(), i4_3g2a_snapshot=g2a)


def test_provider_evidence_requires_all_g1_required_metrics() -> None:
    metrics = {
        "execution_completed": True,
        "source_input_identity_preserved": True,
        "authoritative_sfm_preserved": True,
        "adaptive_plan_bound": True,
        "provider_provenance_complete": True,
        "forbidden_authority_claims_absent": True,
        "wall_time_s": 1.0,
        "peak_vram_mib": 512.0,
        "selected_view_count": 5,
        "processed_view_count": 5,
        "aligned_view_fraction": 1.0,
    }
    optional = {
        key: "not available from frozen adapter"
        for key in (
            "peak_host_memory_mib", "sparse_correspondence_coverage", "alignment_median_relative_residual",
            "alignment_p95_relative_residual", "inter_view_consistency_ratio", "inter_view_median_relative_residual",
            "registered_view_surface_coverage", "canonical_point_count", "detached_component_fraction",
            "unsupported_geometry_fraction",
        )
    }
    manifest = {
        "schema_version": 1,
        "scenario_id": "g2_complete_room_mvs_da3",
        "provider_implementation": "colmap_mvs",
        "comparison_mode": "production_native",
        "attempt": 1,
        "status": "pass",
        "source_identity": {},
        "authoritative_sfm_identity": {},
        "adaptive_plan_identity": {},
        "provider_provenance": {},
        "runtime_resources": {},
        "metrics": metrics,
        "optional_metric_reasons": optional,
        "artifacts": [],
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "provider_promoted": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
    }
    validate_provider_run_evidence(manifest)
    del manifest["metrics"]["peak_vram_mib"]
    with pytest.raises(ReconstructionValidationError, match="Required G1 metric missing"):
        validate_provider_run_evidence(manifest)
