from __future__ import annotations

import json
from pathlib import Path

from core.reconstruction.dense_provider_architecture import DenseProviderRunStatus
from core.reconstruction.mvs_provider import MvsProductionProviderAdapter, MvsProviderRunContext
from tests.i4_3b_helpers import CollectingProgressSink, FakeWorkerRunner, explicit_selection, good_environment, good_resources, make_sfm_result


def test_i43b_end_to_end_contract_writes_auditable_summary_without_changing_inputs(tmp_path: Path) -> None:
    sfm = make_sfm_result(tmp_path, view_count=5)
    py = tmp_path / "python3.11"
    worker = tmp_path / "worker.py"
    py.write_text("", encoding="utf-8")
    worker.write_text("", encoding="utf-8")
    context = MvsProviderRunContext(
        sfm_result=sfm,
        reconstruction_frame_id="canonical-sfm-frame",
        output_dir=tmp_path / "provider-output",
        pycolmap_python=py,
        worker_script=worker,
    )
    sink = CollectingProgressSink()
    adapter = MvsProductionProviderAdapter(
        context,
        progress_sink=sink,
        environment_probe=good_environment(str(py)),
        resource_probe=good_resources(context.output_dir),
        worker_runner=FakeWorkerRunner(ply_points=7),
    )
    request, decision = explicit_selection(5, frame_id="canonical-sfm-frame")
    result = adapter.execute(request, decision)
    assert result.status is DenseProviderRunStatus.COMPLETED
    summary = json.loads((context.output_dir / "i4_3b_summary.json").read_text(encoding="utf-8"))
    worker_request = json.loads((context.output_dir / "worker/request.json").read_text(encoding="utf-8"))
    assert worker_request["expected_registered_views"] == 5
    assert len(worker_request["image_names"]) == 5
    assert worker_request["config"]["all_registered_views_selected"] is True
    assert worker_request["config"]["permanent_global_view_cap"] is None
    assert worker_request["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert worker_request["authority"]["sfm_rerun"] is False
    assert summary["status"] == "pass"
    assert summary["integrity"]["authoritative_sparse_unchanged"] is True
    assert summary["integrity"]["registered_images_unchanged"] is True
    assert summary["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert summary["authority"]["production_filter_policy_changed"] is False
    assert summary["authority"]["provider_promotion_performed"] is False
    assert summary["execution_result"]["selected_view_count"] == 5
    assert summary["execution_result"]["permanent_global_view_cap"] is None
    assert summary["provider"]["qualification_state"] == "candidate"
    assert len(sink.events) >= 6


def test_i43b_isolated_worker_retains_the_qualified_mvs_and_sparse_invariance_gates() -> None:
    worker = Path("scripts/reconstruction_pycolmap_dense_provider_mvs_worker.py")
    source = worker.read_text(encoding="utf-8")
    assert '"sparse_point_ids_preserved"' in source
    assert '"sparse_point_positions_preserved"' in source
    assert "patch_options.geom_consistency = True" in source
    assert "patch_options.filter = True" in source
    assert 'input_type="geometric"' in source
    assert "fused.ply.vis" in source or "vis_path" in source
    assert "incremental_mapping" not in source
    assert "pycolmap.bundle" not in source
    assert "pose_refinement" in source  # locked false authority field, never an operation
