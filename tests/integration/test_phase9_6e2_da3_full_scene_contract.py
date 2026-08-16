from __future__ import annotations

from pathlib import Path


def test_e2_source_contract_preserves_quick_da3_and_uses_e1_supervisor() -> None:
    root = Path(__file__).resolve().parents[2]
    page = (root / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    quick = (root / "app/da3_canonical_integration_controller.py").read_text(encoding="utf-8")
    full = (root / "app/reconstruction_da3_full_scene.py").read_text(encoding="utf-8")
    worker = (root / "scripts/run_phase9_6e2_full_scene_da3_job.py").read_text(encoding="utf-8")
    assert 'QPushButton("Integrate DA3")' in page
    assert 'QPushButton("Build Full Scene")' in page
    assert "class DA3CanonicalIntegrationController" in quick
    assert "build_da3_canonical_integration_command" in quick
    assert "start_digital_twin_processing_job(spec, command)" in page
    assert "DigitalTwinProcessingKind.DENSE_GEOMETRY" in full
    assert "metric_scale_authoritative\": False" in worker
    assert '"loop_closure": False' in worker
    assert "unit_to_metre\": None" in worker
