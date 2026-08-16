from __future__ import annotations

from pathlib import Path

from app.reconstruction_nerf_build import NERF_BUILD_PROTOCOL


ROOT = Path(__file__).resolve().parents[2]


def test_e3_direct_worker_is_e1_protocol_bound_and_does_not_spawn_nested_trainer() -> None:
    source = (ROOT / "scripts/nerf_workers/phase9_6e3_nerf_build_worker.py").read_text(encoding="utf-8")
    assert NERF_BUILD_PROTOCOL in source
    assert "run_qualified_nerfstudio_worker(inner_request, inner_result)" in source
    assert "subprocess.Popen" not in source
    assert '"canonical_geometry_authoritative": False' in source
    assert '"metric_scale_authoritative": False' in source
    assert '"measurement_authoritative": False' in source
    assert '"qc_authoritative": False' in source
    assert '"certified_metrology": False' in source


def test_e3_page_wires_build_and_rebuild_through_generic_e1_supervisor() -> None:
    source = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    assert "Build & Train NeRF" in source
    assert "Rebuild NeRF" in source
    assert "build_nerf_training_job(" in source
    assert "self.start_digital_twin_processing_job(spec, command)" in source
    assert "NERF_BUILD_PROTOCOL" in source
    assert "Active canonical scene changed while NeRF was training" in source
    assert "self._controller.load_persistent_scene(manifest, verify_file_references=True)" in source
