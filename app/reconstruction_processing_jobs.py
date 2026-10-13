"""Dependency-light Digital Twin processing-job contracts for Phase 9.6E1.

This module does not launch DA3, Gaussian, Nerfstudio, CUDA, or any other heavy
backend.  It defines the UI/application-side lifecycle and provenance contract
that later Phase 9.6E2/E3 adapters use while the actual heavy work remains in
qualified external runtimes.

The contract deliberately permits at most one heavyweight Digital Twin build
job at a time and never grants geometry, scale, measurement, QC, or metrology
authority to a processing job.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Mapping
from uuid import UUID



RESERVED_DIGITAL_TWIN_ENV_KEYS = frozenset({
    "DEPTH3D_DIGITAL_TWIN_JOB_ID",
    "DEPTH3D_DIGITAL_TWIN_EXPECTED_PROTOCOL",
    "DEPTH3D_DIGITAL_TWIN_OUTPUT_DIR",
})


class DigitalTwinProcessingKind(str, Enum):
    """Product-facing class of a long-running Digital Twin build job."""

    DENSE_GEOMETRY = "dense_geometry"
    GAUSSIAN_APPEARANCE = "gaussian_appearance"
    NERF_APPEARANCE = "nerf_appearance"


class DigitalTwinProcessingLifecycle(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_active(self) -> bool:
        return self in {
            DigitalTwinProcessingLifecycle.PREPARING,
            DigitalTwinProcessingLifecycle.RUNNING,
            DigitalTwinProcessingLifecycle.CANCELLING,
        }

    @property
    def is_terminal(self) -> bool:
        return self in {
            DigitalTwinProcessingLifecycle.COMPLETED,
            DigitalTwinProcessingLifecycle.FAILED,
            DigitalTwinProcessingLifecycle.CANCELLED,
        }


@dataclass(frozen=True, slots=True)
class DigitalTwinExternalProcessPolicy:
    """Non-negotiable execution policy for future E2/E3 build adapters."""

    shell: bool = False
    start_new_session: bool = True
    max_active_heavy_jobs: int = 1
    termination_order: tuple[str, str] = ("SIGTERM", "SIGKILL")
    canonical_fallback_required: bool = True
    heavy_backends_external_only: bool = True
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        if self.shell:
            raise ValueError("Digital Twin processing jobs must never use shell=True.")
        if not self.start_new_session:
            raise ValueError("Digital Twin processing jobs require an isolated process group/session.")
        if int(self.max_active_heavy_jobs) != 1:
            raise ValueError("Phase 9.6E1 permits exactly one active heavyweight build job.")
        if tuple(self.termination_order) != ("SIGTERM", "SIGKILL"):
            raise ValueError("Digital Twin processing cancellation must be SIGTERM then SIGKILL.")
        if not self.canonical_fallback_required:
            raise ValueError("Canonical fallback must remain available during processing failures.")
        if not self.heavy_backends_external_only:
            raise ValueError("Heavy Digital Twin backends must remain external to the depth environment.")
        if self.certified_metrology:
            raise ValueError("A processing job cannot certify metrology.")


@dataclass(frozen=True, slots=True)
class DigitalTwinProcessingJobSpec:
    """Immutable identity/provenance for one long-running build job."""

    job_id: str
    kind: DigitalTwinProcessingKind
    title: str
    backend_label: str
    runtime_label: str
    output_dir: Path
    expected_protocol: str
    scene_id: str = ""
    source_label: str = ""
    profile_label: str = ""
    cancellable: bool = True

    def __post_init__(self) -> None:
        try:
            canonical_job_id = str(UUID(str(self.job_id).strip()))
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValueError("Digital Twin processing job_id must be a valid UUID.") from exc
        if not isinstance(self.kind, DigitalTwinProcessingKind):
            raise TypeError("kind must be DigitalTwinProcessingKind.")
        title = str(self.title).strip()
        backend = str(self.backend_label).strip()
        runtime = str(self.runtime_label).strip()
        protocol = str(self.expected_protocol).strip()
        if not title:
            raise ValueError("Digital Twin processing title is required.")
        if not backend:
            raise ValueError("Digital Twin processing backend_label is required.")
        if not runtime:
            raise ValueError("Digital Twin processing runtime_label is required.")
        if not protocol:
            raise ValueError("Digital Twin processing expected_protocol is required.")
        output = Path(self.output_dir).expanduser().resolve()
        object.__setattr__(self, "job_id", canonical_job_id)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "backend_label", backend)
        object.__setattr__(self, "runtime_label", runtime)
        object.__setattr__(self, "output_dir", output)
        object.__setattr__(self, "expected_protocol", protocol)
        object.__setattr__(self, "scene_id", str(self.scene_id).strip())
        object.__setattr__(self, "source_label", str(self.source_label).strip())
        object.__setattr__(self, "profile_label", str(self.profile_label).strip())
        object.__setattr__(self, "cancellable", bool(self.cancellable))

    @property
    def authority_text(self) -> str:
        if self.kind is DigitalTwinProcessingKind.DENSE_GEOMETRY:
            return "Supporting geometry build only • canonical authority unchanged"
        return "Appearance build only • canonical authority unchanged"


@dataclass(frozen=True, slots=True)
class DigitalTwinProcessingCommand:
    """Shell-free command definition consumed by the Qt-side E1 supervisor."""

    program: Path
    arguments: tuple[str, ...]
    working_directory: Path
    environment_overrides: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        program = Path(self.program).expanduser().resolve()
        working = Path(self.working_directory).expanduser().resolve()
        if not program.is_file():
            raise ValueError(f"Digital Twin processing executable is unavailable: {program}")
        if not working.is_dir():
            raise ValueError(f"Digital Twin processing working directory is unavailable: {working}")
        arguments = tuple(str(item) for item in self.arguments)
        overrides: list[tuple[str, str]] = []
        for key, value in self.environment_overrides:
            name = str(key).strip()
            if not name or "=" in name or "\x00" in name:
                raise ValueError("Invalid processing environment variable name.")
            if name in RESERVED_DIGITAL_TWIN_ENV_KEYS:
                raise ValueError(f"Processing environment override is reserved by E1: {name}")
            overrides.append((name, str(value)))
        object.__setattr__(self, "program", program)
        object.__setattr__(self, "working_directory", working)
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(self, "environment_overrides", tuple(overrides))


@dataclass(frozen=True, slots=True)
class DigitalTwinProcessingSnapshot:
    """Immutable current state surfaced by the E1 job supervisor."""

    lifecycle: DigitalTwinProcessingLifecycle
    spec: DigitalTwinProcessingJobSpec | None = None
    stage: str = ""
    message: str = ""
    progress_fraction: float | None = None
    elapsed_seconds: float = 0.0
    completed_units: int | None = None
    total_units: int | None = None
    result_path: str = ""
    error: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.lifecycle, DigitalTwinProcessingLifecycle):
            raise TypeError("lifecycle must be DigitalTwinProcessingLifecycle.")
        if self.lifecycle is DigitalTwinProcessingLifecycle.IDLE and self.spec is not None:
            raise ValueError("Idle processing snapshot cannot carry a job spec.")
        if self.lifecycle is not DigitalTwinProcessingLifecycle.IDLE and self.spec is None:
            raise ValueError("Non-idle processing snapshot requires a job spec.")
        fraction = self.progress_fraction
        if fraction is not None:
            fraction = float(fraction)
            if not isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                raise ValueError("progress_fraction must be finite within [0, 1].")
            object.__setattr__(self, "progress_fraction", fraction)
        elapsed = float(self.elapsed_seconds)
        if not isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("elapsed_seconds must be finite and non-negative.")
        object.__setattr__(self, "elapsed_seconds", elapsed)
        completed = self.completed_units
        total = self.total_units
        if completed is not None and int(completed) < 0:
            raise ValueError("completed_units must be non-negative.")
        if total is not None and int(total) <= 0:
            raise ValueError("total_units must be positive.")
        if completed is not None:
            object.__setattr__(self, "completed_units", int(completed))
        if total is not None:
            object.__setattr__(self, "total_units", int(total))
        if completed is not None and total is not None and int(completed) > int(total):
            raise ValueError("completed_units cannot exceed total_units.")
        object.__setattr__(self, "stage", str(self.stage).strip())
        object.__setattr__(self, "message", str(self.message).strip())
        object.__setattr__(self, "result_path", str(self.result_path).strip())
        object.__setattr__(self, "error", str(self.error).strip())

    @property
    def is_active(self) -> bool:
        return self.lifecycle.is_active

    @property
    def can_cancel(self) -> bool:
        return bool(
            self.spec is not None
            and self.spec.cancellable
            and self.lifecycle in {
                DigitalTwinProcessingLifecycle.PREPARING,
                DigitalTwinProcessingLifecycle.RUNNING,
            }
        )

    @property
    def job_text(self) -> str:
        if self.spec is None:
            return "No Digital Twin build job"
        return self.spec.title

    @property
    def lifecycle_text(self) -> str:
        return self.lifecycle.value.replace("_", " ").title()

    @property
    def stage_text(self) -> str:
        if self.lifecycle is DigitalTwinProcessingLifecycle.IDLE:
            return "Waiting for build request"
        return self.stage.replace("_", " ").title() if self.stage else self.lifecycle_text

    @property
    def progress_text(self) -> str:
        if self.lifecycle is DigitalTwinProcessingLifecycle.IDLE:
            return "—"
        if self.completed_units is not None and self.total_units is not None:
            suffix = f"{self.completed_units} / {self.total_units}"
            if self.progress_fraction is not None:
                return f"{round(self.progress_fraction * 100.0):d}% • {suffix}"
            return suffix
        if self.progress_fraction is not None:
            return f"{round(self.progress_fraction * 100.0):d}%"
        return "Active" if self.is_active else self.lifecycle_text

    @property
    def backend_text(self) -> str:
        if self.spec is None:
            return "External runtime not active"
        profile = f" • {self.spec.profile_label}" if self.spec.profile_label else ""
        return f"{self.spec.backend_label} • {self.spec.runtime_label}{profile}"

    @property
    def output_text(self) -> str:
        if self.result_path:
            return Path(self.result_path).name or self.result_path
        if self.spec is not None:
            return self.spec.output_dir.name or str(self.spec.output_dir)
        return "—"

    @property
    def authority_text(self) -> str:
        if self.spec is None:
            return "Canonical geometry remains authoritative"
        return self.spec.authority_text


DEFAULT_DIGITAL_TWIN_PROCESS_POLICY = DigitalTwinExternalProcessPolicy()


def idle_digital_twin_processing_snapshot() -> DigitalTwinProcessingSnapshot:
    return DigitalTwinProcessingSnapshot(DigitalTwinProcessingLifecycle.IDLE)


def begin_digital_twin_processing_snapshot(
    spec: DigitalTwinProcessingJobSpec,
) -> DigitalTwinProcessingSnapshot:
    if not isinstance(spec, DigitalTwinProcessingJobSpec):
        raise TypeError("spec must be DigitalTwinProcessingJobSpec.")
    return DigitalTwinProcessingSnapshot(
        DigitalTwinProcessingLifecycle.PREPARING,
        spec=spec,
        stage="preparing",
        message="Preparing isolated external processing job.",
    )


def update_digital_twin_processing_snapshot(
    snapshot: DigitalTwinProcessingSnapshot,
    *,
    lifecycle: DigitalTwinProcessingLifecycle | None = None,
    stage: str | None = None,
    message: str | None = None,
    progress_fraction: float | None = None,
    elapsed_seconds: float | None = None,
    completed_units: int | None = None,
    total_units: int | None = None,
    result_path: str | None = None,
    error: str | None = None,
) -> DigitalTwinProcessingSnapshot:
    if not isinstance(snapshot, DigitalTwinProcessingSnapshot):
        raise TypeError("snapshot must be DigitalTwinProcessingSnapshot.")
    if snapshot.spec is None:
        raise ValueError("Cannot update an idle Digital Twin processing snapshot.")
    next_lifecycle = lifecycle or snapshot.lifecycle
    if snapshot.lifecycle.is_terminal and next_lifecycle is not snapshot.lifecycle:
        raise ValueError("Terminal Digital Twin processing snapshots cannot transition again.")
    return replace(
        snapshot,
        lifecycle=next_lifecycle,
        stage=snapshot.stage if stage is None else str(stage),
        message=snapshot.message if message is None else str(message),
        progress_fraction=snapshot.progress_fraction if progress_fraction is None else progress_fraction,
        elapsed_seconds=snapshot.elapsed_seconds if elapsed_seconds is None else elapsed_seconds,
        completed_units=snapshot.completed_units if completed_units is None else completed_units,
        total_units=snapshot.total_units if total_units is None else total_units,
        result_path=snapshot.result_path if result_path is None else str(result_path),
        error=snapshot.error if error is None else str(error),
    )


def apply_digital_twin_processing_event(
    snapshot: DigitalTwinProcessingSnapshot,
    payload: Mapping[str, object],
    *,
    elapsed_seconds: float,
) -> DigitalTwinProcessingSnapshot:
    """Apply one normalized external-worker event without granting authority.

    The process supervisor remains responsible for terminal state based on the
    actual child-process exit.  Worker events update stage/progress/result hints
    only; a backend cannot self-promote its own job to an authoritative result.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping.")
    event = str(payload.get("event", payload.get("type", "status"))).strip().lower()
    stage = str(payload.get("stage", snapshot.stage)).strip()
    message = str(payload.get("message", snapshot.message)).strip()
    fraction_raw = payload.get("progress_fraction", None)
    fraction = snapshot.progress_fraction if fraction_raw is None else float(fraction_raw)
    completed_raw = payload.get("completed_units", None)
    total_raw = payload.get("total_units", None)
    completed = snapshot.completed_units if completed_raw is None else int(completed_raw)
    total = snapshot.total_units if total_raw is None else int(total_raw)
    result_path = str(payload.get("result_path", snapshot.result_path)).strip()
    error = str(payload.get("error", snapshot.error)).strip()
    lifecycle = snapshot.lifecycle
    if lifecycle is DigitalTwinProcessingLifecycle.PREPARING and event in {
        "started",
        "status",
        "progress",
        "ready",
    }:
        lifecycle = DigitalTwinProcessingLifecycle.RUNNING
    return update_digital_twin_processing_snapshot(
        snapshot,
        lifecycle=lifecycle,
        stage=stage,
        message=message,
        progress_fraction=fraction,
        elapsed_seconds=elapsed_seconds,
        completed_units=completed,
        total_units=total,
        result_path=result_path,
        error=error,
    )


__all__ = [
    "DEFAULT_DIGITAL_TWIN_PROCESS_POLICY",
    "RESERVED_DIGITAL_TWIN_ENV_KEYS",
    "DigitalTwinExternalProcessPolicy",
    "DigitalTwinProcessingCommand",
    "DigitalTwinProcessingJobSpec",
    "DigitalTwinProcessingKind",
    "DigitalTwinProcessingLifecycle",
    "DigitalTwinProcessingSnapshot",
    "apply_digital_twin_processing_event",
    "begin_digital_twin_processing_snapshot",
    "idle_digital_twin_processing_snapshot",
    "update_digital_twin_processing_snapshot",
]
