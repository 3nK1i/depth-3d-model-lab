from core.reconstruction import ReconstructionPipelineRunStatus, summarize_f3_pipeline_result
from tests.reconstruction_f3_helpers import make_f3_fixture


def test_repeated_f3_runs_keep_session_payloads_isolated_and_deterministic():
    signatures = []
    session_ids = set()
    for index in range(3):
        fixture = make_f3_fixture(request_id=f"phase8-7f3-stress-{index}")
        assert fixture.session_id not in session_ids
        session_ids.add(fixture.session_id)
        result = fixture.orchestrator.run()
        assert result.status is ReconstructionPipelineRunStatus.COMPLETED
        summary = summarize_f3_pipeline_result(result, store=fixture.store)
        signatures.append(
            (
                summary.dense_point_count,
                summary.mesh_vertex_count,
                summary.mesh_face_count,
                summary.semantic_object_count,
                summary.inspection_decision,
                summary.metric_geometry_status,
            )
        )
    assert len(set(signatures)) == 1
