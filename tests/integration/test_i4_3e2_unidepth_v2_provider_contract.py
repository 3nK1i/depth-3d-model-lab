from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.reconstruction.unidepth_v2_dense_provider import UniDepthV2ProductionProviderAdapter
from tests.i4_3e2_helpers import FakeUniDepthWorkerRunner, explicit_unidepth_selection, make_unidepth_context


def test_i43e2_unidepth_execution_persists_canonical_candidate_and_native_evidence(tmp_path: Path) -> None:
    context, env = make_unidepth_context(tmp_path, view_count=4)
    before = {
        item.keyframe_id: (Path(context.sfm_result.workspace.workspace_root) / item.staged_image_path).read_bytes()
        for item in context.sfm_result.workspace.images
    }
    adapter = UniDepthV2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeUniDepthWorkerRunner(uncertainty_value=0.5),
    )
    request, decision = explicit_unidepth_selection(4)
    result = adapter.execute(request, decision)

    canonical = Path(result.canonical_geometry_ref)
    assert canonical.is_file()
    with np.load(canonical, allow_pickle=False) as archive:
        assert archive["points_reconstruction"].ndim == 2
        assert archive["points_reconstruction"].shape[1] == 3
        assert len(archive["points_reconstruction"]) > 0
        assert np.isfinite(archive["mean_confidence"]).all()
        assert np.allclose(archive["mean_confidence"], 2.0 / 3.0, atol=1e-5)

    summary_path = context.output_dir / "i4_3e2_unidepth_v2_execution_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["provider"]["family"] == "single_image_depth"
    assert summary["provider"]["metadata"]["native_confidence_kind"] == "uncertainty"
    assert summary["execution"]["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert summary["authority"]["provider_promoted"] is False
    assert summary["authority"]["model_direct_points_consumed_for_canonical_geometry"] is False
    assert summary["config"]["permanent_global_view_cap"] is None
    assert summary["config"]["adaptive_view_selection_deferred_to"] == "I4.3F"
    assert all(record["native_uncertainty_transformed_to_fusion_confidence"] for record in summary["alignment"]["records"])

    worker_manifest = Path(summary["execution"]["native_artifact_refs"][0])
    worker = json.loads(worker_manifest.read_text(encoding="utf-8"))
    first_evidence = Path(worker["frame_records"][0]["evidence_npz_path"])
    with np.load(first_evidence, allow_pickle=False) as archive:
        assert "direct_points_model" in archive.files
        assert "model_intrinsics" in archive.files
        assert "uncertainty" in archive.files

    for item in context.sfm_result.workspace.images:
        path = Path(context.sfm_result.workspace.workspace_root) / item.staged_image_path
        assert path.read_bytes() == before[item.keyframe_id]
