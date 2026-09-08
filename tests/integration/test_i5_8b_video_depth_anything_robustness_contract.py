from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from core.reconstruction.i5_video_depth_anything_robustness_replay import analyze_i5_8b_provider_summary, build_i5_8b_video_depth_anything_robustness_replay
from core.reconstruction.video_depth_anything_dense_provider import VideoDepthAnythingProductionProviderAdapter
from tests.i4_3d1_helpers import FakeVDAWorkerRunner, explicit_vda_selection, make_vda_context
from tests.i5_8b_helpers import build_args


def test_i58b_analyzer_accepts_actual_i43d1_execution_summary_shape(tmp_path: Path) -> None:
    context, environment = make_vda_context(tmp_path, view_count=4)
    resolved_runtime_config = dict(context.runtime_config)
    resolved_runtime_config["source_dir"] = environment.source_directory
    resolved_runtime_config["checkpoint"] = environment.checkpoint_path
    context = replace(context, runtime_config=resolved_runtime_config)
    adapter = VideoDepthAnythingProductionProviderAdapter(context=context, environment=environment, runner=FakeVDAWorkerRunner())
    request, decision = explicit_vda_selection(4)
    adapter.execute(request, decision)
    summary = json.loads((context.output_dir / "i4_3d1_vda_execution_summary.json").read_text(encoding="utf-8"))
    # The synthetic I4.3D1 fake worker predates B's required real latency evidence; add only
    # fields the real production worker already persists.
    for i, record in enumerate(summary["worker"]["frame_records"]):
        record.update({"timestamp_ns": (i + 1) * 1000000, "latency_ms": 10.0 + i, "depth_scale_native": "metric", "intrinsics_source_native": "estimated"})
        record["keyframe_id"] = summary["temporal_sequence"]["ordered_keyframe_ids"][i]
    summary["worker"]["elapsed_s"] = 0.2
    summary["worker"]["authority"] = {
        "native_metric_depth_preserved_as_evidence": True,
        "native_estimated_intrinsics_preserved_as_evidence": True,
        "native_metric_depth_promoted_to_canonical_metric_scale": False,
        "native_estimated_intrinsics_promoted_to_canonical_projection": False,
        "camera_pose_authority": "pycolmap_sfm",
    }
    analysis = analyze_i5_8b_provider_summary(
        summary,
        wall_time_s=0.5,
        resource_observation={"peak_vram_mib": None, "peak_system_ram_used_mib": None},
        expected_runtime_config=dict(context.runtime_config),
    )
    assert analysis["runtime_status"] == "pass"
    assert analysis["native_metrics"]["processed_views"] == 4
    assert analysis["native_metrics"]["frame_latency_count"] == 4


def test_i58b_full_contract_preserves_i58a_and_authorizes_only_i58c() -> None:
    out = build_i5_8b_video_depth_anything_robustness_replay(**build_args())
    assert out["decision"]["i5_8a_complete"] is True
    assert out["decision"]["i5_8b_complete"] is True
    assert out["decision"]["i5_8c_authorized"] is True
    assert out["decision"]["i5_9_started"] is False
    assert out["scope"]["sfm_rerun"] is False
    assert out["scope"]["mvs_rerun"] is False
    assert out["scope"]["retry_until_pass"] is False
