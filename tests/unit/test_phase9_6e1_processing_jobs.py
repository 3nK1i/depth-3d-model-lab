from __future__ import annotations

from pathlib import Path
import sys
from uuid import uuid4

import pytest

from app.reconstruction_processing_jobs import (
    DEFAULT_DIGITAL_TWIN_PROCESS_POLICY,
    DigitalTwinExternalProcessPolicy,
    DigitalTwinProcessingCommand,
    DigitalTwinProcessingJobSpec,
    DigitalTwinProcessingKind,
    DigitalTwinProcessingLifecycle,
    apply_digital_twin_processing_event,
    begin_digital_twin_processing_snapshot,
    idle_digital_twin_processing_snapshot,
    update_digital_twin_processing_snapshot,
)


def _spec(tmp_path: Path, *, kind: DigitalTwinProcessingKind = DigitalTwinProcessingKind.NERF_APPEARANCE):
    return DigitalTwinProcessingJobSpec(
        job_id=str(uuid4()),
        kind=kind,
        title="Build NeRF Appearance" if kind is DigitalTwinProcessingKind.NERF_APPEARANCE else "Build Dense Geometry",
        backend_label="Nerfstudio Nerfacto" if kind is DigitalTwinProcessingKind.NERF_APPEARANCE else "Depth Anything 3",
        runtime_label="external-test-runtime",
        output_dir=tmp_path / "out",
        expected_protocol="eia.digital-twin.processing.test.v1",
        scene_id="scene-e1",
        source_label="fixture",
        profile_label="Balanced",
    )


def test_e1_policy_locks_external_shell_free_single_job_lifecycle() -> None:
    policy = DEFAULT_DIGITAL_TWIN_PROCESS_POLICY
    assert policy.shell is False
    assert policy.start_new_session is True
    assert policy.max_active_heavy_jobs == 1
    assert policy.termination_order == ("SIGTERM", "SIGKILL")
    assert policy.canonical_fallback_required is True
    assert policy.heavy_backends_external_only is True
    assert policy.certified_metrology is False

    with pytest.raises(ValueError, match="shell=True"):
        DigitalTwinExternalProcessPolicy(shell=True)
    with pytest.raises(ValueError, match="exactly one"):
        DigitalTwinExternalProcessPolicy(max_active_heavy_jobs=2)


def test_e1_job_spec_keeps_build_authority_noncanonical(tmp_path: Path) -> None:
    dense = _spec(tmp_path, kind=DigitalTwinProcessingKind.DENSE_GEOMETRY)
    nerf = _spec(tmp_path, kind=DigitalTwinProcessingKind.NERF_APPEARANCE)
    assert "Supporting geometry" in dense.authority_text
    assert "canonical authority unchanged" in dense.authority_text
    assert "Appearance build only" in nerf.authority_text
    assert "canonical authority unchanged" in nerf.authority_text


def test_e1_command_is_explicit_shell_free_process_input(tmp_path: Path) -> None:
    command = DigitalTwinProcessingCommand(
        program=Path(sys.executable),
        arguments=("-c", "print('ok')"),
        working_directory=tmp_path,
        environment_overrides=(("EIA_TEST", "1"),),
    )
    assert command.program.is_file()
    assert command.arguments[0] == "-c"
    assert command.environment_overrides == (("EIA_TEST", "1"),)
    with pytest.raises(ValueError, match="reserved by E1"):
        DigitalTwinProcessingCommand(
            program=Path(sys.executable),
            arguments=("-c", "print('bad')"),
            working_directory=tmp_path,
            environment_overrides=(("EIA_DIGITAL_TWIN_JOB_ID", "spoofed"),),
        )


def test_e1_idle_and_begin_snapshots_present_truthful_state(tmp_path: Path) -> None:
    idle = idle_digital_twin_processing_snapshot()
    assert idle.lifecycle is DigitalTwinProcessingLifecycle.IDLE
    assert idle.job_text == "No Digital Twin build job"
    assert idle.authority_text == "Canonical geometry remains authoritative"

    started = begin_digital_twin_processing_snapshot(_spec(tmp_path))
    assert started.lifecycle is DigitalTwinProcessingLifecycle.PREPARING
    assert started.can_cancel is True
    assert started.progress_fraction is None
    assert "canonical authority unchanged" in started.authority_text


def test_e1_progress_event_updates_stage_fraction_counts_and_elapsed(tmp_path: Path) -> None:
    snapshot = begin_digital_twin_processing_snapshot(_spec(tmp_path))
    updated = apply_digital_twin_processing_event(
        snapshot,
        {
            "event": "progress",
            "stage": "optimization",
            "message": "Optimizing external representation.",
            "progress_fraction": 0.425,
            "completed_units": 425,
            "total_units": 1000,
        },
        elapsed_seconds=12.5,
    )
    assert updated.lifecycle is DigitalTwinProcessingLifecycle.RUNNING
    assert updated.stage == "optimization"
    assert updated.progress_fraction == pytest.approx(0.425)
    assert updated.completed_units == 425
    assert updated.total_units == 1000
    assert updated.elapsed_seconds == pytest.approx(12.5)
    assert updated.progress_text == "42% • 425 / 1000"


def test_e1_worker_event_cannot_self_promote_to_terminal_authoritative_result(tmp_path: Path) -> None:
    snapshot = begin_digital_twin_processing_snapshot(_spec(tmp_path))
    updated = apply_digital_twin_processing_event(
        snapshot,
        {
            "event": "completed",
            "stage": "backend_done",
            "message": "Backend says done",
            "progress_fraction": 1.0,
            "canonical_geometry_authoritative": True,
            "certified_metrology": True,
        },
        elapsed_seconds=3.0,
    )
    # Pure event projection never accepts terminal authority; the Qt supervisor
    # validates unsafe authority claims and actual process exit separately.
    assert updated.lifecycle is DigitalTwinProcessingLifecycle.PREPARING
    assert updated.progress_fraction == 1.0
    assert updated.spec is not None
    assert "canonical authority unchanged" in updated.authority_text


def test_e1_terminal_snapshot_cannot_transition_back_to_running(tmp_path: Path) -> None:
    snapshot = begin_digital_twin_processing_snapshot(_spec(tmp_path))
    completed = update_digital_twin_processing_snapshot(
        snapshot,
        lifecycle=DigitalTwinProcessingLifecycle.COMPLETED,
        progress_fraction=1.0,
        elapsed_seconds=7.0,
    )
    assert completed.lifecycle is DigitalTwinProcessingLifecycle.COMPLETED
    assert completed.can_cancel is False
    with pytest.raises(ValueError, match="Terminal"):
        update_digital_twin_processing_snapshot(
            completed,
            lifecycle=DigitalTwinProcessingLifecycle.RUNNING,
        )
