from __future__ import annotations

from pathlib import Path

from core.benchmark.events import (
    EVENT_PREFIX,
    encode_event,
    parse_event_line,
)
from core.reconstruction.reconstruction_progressive_sfm_transport import (
    PROGRESSIVE_SFM_EVENT_PREFIX,
)


ROOT = Path(__file__).resolve().parents[2]


def test_b4_line_protocols_use_public_namespace() -> None:
    assert EVENT_PREFIX == "DEPTH3D_BENCHMARK_EVENT "
    line = encode_event("namespace_probe", value=1)
    assert line.startswith(EVENT_PREFIX)
    assert parse_event_line(line) == {
        "event": "namespace_probe",
        "value": 1,
    }

    assert (
        PROGRESSIVE_SFM_EVENT_PREFIX
        == "DEPTH3D_SFM_PROGRESSIVE "
    )


def test_b4_nerf_interactive_protocol_uses_public_prefix() -> None:
    for relative in (
        "app/widgets/reconstruction_nerf_worker_client.py",
        "scripts/nerf_workers/nerfstudio_interactive_worker.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert 'PROTOCOL_PREFIX = "DEPTH3D_NERF_JSON "' in source


def test_b4_unpinned_process_protocols_use_public_namespace() -> None:
    expected = {
        "app/da3_canonical_integration_controller.py":
            "depth3d.reconstruction.da3-canonical.v1",
        "scripts/run_phase9_3h_da3_canonical_job.py":
            "depth3d.reconstruction.da3-canonical.v1",
        "app/reconstruction_da3_full_scene.py":
            "depth3d.digital-twin.da3-full-scene.v1",
        "app/reconstruction_nerf_build.py":
            "depth3d.digital-twin.nerf-build.v1",
        "scripts/nerf_workers/phase9_6e3_nerf_build_worker.py":
            "depth3d.digital-twin.nerf-build.v1",
        "app/reconstruction_dense_job_controller.py":
            "depth3d.reconstruction.dense.post_sparse.v1",
        "scripts/run_i6_7e2b2a_post_sparse_dense_worker.py":
            "depth3d.reconstruction.dense.post_sparse.v1",
        "scripts/run_i6_7e_dense_completion_verify.py":
            "depth3d.reconstruction.dense.verify.v1",
    }

    for relative, protocol in expected.items():
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert protocol in source
