from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.reconstruction_processing_jobs import (
    DigitalTwinProcessingJobSpec,
    DigitalTwinProcessingKind,
    DigitalTwinProcessingLifecycle,
    begin_digital_twin_processing_snapshot,
)


def test_e4_focus_uses_existing_e1_job_identity_without_new_authority(tmp_path: Path) -> None:
    spec = DigitalTwinProcessingJobSpec(
        job_id=str(uuid4()),
        kind=DigitalTwinProcessingKind.NERF_APPEARANCE,
        title="Rebuild NeRF",
        backend_label="Nerfstudio Nerfacto",
        runtime_label="external eia-nerf-nerfstudio",
        output_dir=tmp_path / "nerf",
        expected_protocol="depth3d.digital-twin.nerf-build.v1",
        scene_id="prs-canonical",
    )
    snapshot = begin_digital_twin_processing_snapshot(spec)

    assert snapshot.lifecycle is DigitalTwinProcessingLifecycle.PREPARING
    assert snapshot.spec.job_id == spec.job_id
    assert snapshot.authority_text == "Appearance build only • canonical authority unchanged"
    assert snapshot.can_cancel is True


def test_e4_dense_job_authority_text_remains_supporting_only(tmp_path: Path) -> None:
    spec = DigitalTwinProcessingJobSpec(
        job_id=str(uuid4()),
        kind=DigitalTwinProcessingKind.DENSE_GEOMETRY,
        title="Build Full-Scene DA3",
        backend_label="Depth Anything 3",
        runtime_label="external eia-da3",
        output_dir=tmp_path / "da3",
        expected_protocol="depth3d.digital-twin.da3-full-scene.v1",
    )
    snapshot = begin_digital_twin_processing_snapshot(spec)
    assert snapshot.authority_text == "Supporting geometry build only • canonical authority unchanged"
