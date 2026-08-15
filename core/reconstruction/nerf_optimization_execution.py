"""Phase 9.5C supervised real Nerfstudio/Nerfacto optimization execution.

This module connects the Phase 9.5A qualified external runtime and Phase 9.5B
canonical-scene input bridge to the already-proven Phase 8 Nerfstudio worker.
The production ``depth`` interpreter never imports Nerfstudio or tiny-cuda-nn.
The worker is launched through the qualified external interpreter in its own
process group with bounded timeout and SIGTERM -> SIGKILL escalation.

NeRF remains appearance-only. Canonical reconstruction geometry retains
geometry, scale, picking, measurement, and QC authority. Phase 9.5C proves real
backend execution/checkpoint/render capability only; quality, persistence, UI,
and final resource qualification belong to later Phase 9.5 substeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from typing import Any

from core.contracts.validation import require_nonempty
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.nerf import (
    NerfBackendKind,
    NerfOptimizationProfile,
    NerfSceneArtifact,
    NerfSceneRole,
    NerfTrainingState,
    NerfWorkerExecutionError,
    NerfWorkerRequest,
    nerf_backend_provenance,
)
from core.reconstruction.nerf_interactive_runtime import (
    NerfInteractiveRuntimeReport,
    NerfInteractiveRuntimeState,
    current_process_nerf_heavy_modules,
)
from core.reconstruction.neural_scene import (
    NeuralSceneDataset,
    neural_scene_dataset_fingerprint,
    read_neural_scene_dataset_manifest,
)

PHASE9_5C_NERF_OPTIMIZATION_SCHEMA_VERSION = 1
PHASE9_5C_DEFAULT_TRAINING_STEPS = 64
PHASE9_5C_MAX_TRAINING_STEPS = 256
PHASE9_5C_DEFAULT_MAX_INPUT_FRAMES = 8


class NerfOptimizationExecutionState(str, Enum):
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _isolated_worker_environment(runtime_report: NerfInteractiveRuntimeReport) -> dict[str, str]:
    """Build a child-only conda/CUDA environment without mutating ``depth``."""

    python = Path(runtime_report.spec.python_executable).resolve()
    prefix = python.parent.parent
    if python.parent.name != "bin":
        raise NerfWorkerExecutionError(f"Cannot infer NeRF conda prefix from {python}.")
    env = dict(os.environ)
    env["CONDA_PREFIX"] = str(prefix)
    env["CONDA_DEFAULT_ENV"] = prefix.name
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONPATH", None)
    env["EIA_NERFSTUDIO_ROOT"] = runtime_report.spec.nerfstudio_root
    env["EIA_TCNN_ROOT"] = runtime_report.spec.tiny_cuda_nn_root

    def prepend(key: str, values: list[Path]) -> None:
        parts = [str(value) for value in values if value.is_dir()]
        existing = env.get(key, "")
        if existing:
            parts.append(existing)
        env[key] = os.pathsep.join(parts)

    prepend("PATH", [prefix / "bin"])
    nvcc = prefix / "bin" / "nvcc"
    if nvcc.is_file():
        env["CUDA_HOME"] = str(prefix)
        env["CUDACXX"] = str(nvcc)
    libs = [prefix / "lib", prefix / "lib64", prefix / "targets" / "x86_64-linux" / "lib"]
    prepend("LD_LIBRARY_PATH", libs)
    prepend("LIBRARY_PATH", libs)
    return env


def _terminate_process_group(process: subprocess.Popen[str], *, grace_s: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=max(1.0, grace_s))
    except subprocess.TimeoutExpired:
        pass


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NerfWorkerExecutionError(f"Could not parse {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise NerfWorkerExecutionError(f"{label} must be a JSON object.")
    return payload


@dataclass(frozen=True, slots=True)
class NerfOptimizationExecutionReport:
    state: NerfOptimizationExecutionState
    request_id: str
    runtime_report: NerfInteractiveRuntimeReport
    artifact: NerfSceneArtifact
    runtime_evidence_path: str
    preview_path: str
    worker_console_log_path: str
    optimization_manifest_path: str
    elapsed_seconds: float
    real_backend_started: bool = True
    optimization_completed: bool = True
    rendering_performed: bool = True
    quality_evaluated: bool = False
    performance_qualified: bool = False
    canonical_geometry_authoritative: bool = False
    metric_scale_authoritative: bool = False
    measurement_authoritative: bool = False
    qc_authoritative: bool = False
    certified_metrology: bool = False
    schema_version: int = PHASE9_5C_NERF_OPTIMIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.state is not NerfOptimizationExecutionState.COMPLETED:
            raise ReconstructionValidationError("Successful Phase 9.5C reports must be COMPLETED.")
        object.__setattr__(self, "request_id", require_nonempty(self.request_id, field_name="request_id"))
        if self.runtime_report.state is not NerfInteractiveRuntimeState.READY:
            raise ReconstructionValidationError("Phase 9.5C execution requires a READY 9.5A runtime report.")
        if self.artifact.request_id != self.request_id:
            raise ReconstructionValidationError("NeRF execution artifact request_id mismatch.")
        if self.artifact.training_state is not NerfTrainingState.TRAINED:
            raise ReconstructionValidationError("Phase 9.5C requires a trained NeRF artifact.")
        if self.artifact.scene_role is not NerfSceneRole.APPEARANCE_ONLY:
            raise ReconstructionValidationError("NeRF output must remain appearance-only.")
        if self.artifact.backend_provenance != nerf_backend_provenance(NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL):
            raise ReconstructionValidationError("Phase 9.5C requires the qualified Nerfstudio/Nerfacto backend.")
        elapsed = float(self.elapsed_seconds)
        if not isfinite(elapsed) or elapsed <= 0.0:
            raise ReconstructionValidationError("elapsed_seconds must be finite and positive.")
        for name in (
            "runtime_evidence_path",
            "preview_path",
            "worker_console_log_path",
            "optimization_manifest_path",
        ):
            path = Path(require_nonempty(getattr(self, name), field_name=name))
            if not path.is_absolute():
                raise ReconstructionValidationError(f"{name} must be absolute.")
            object.__setattr__(self, name, str(path))
        if not (self.real_backend_started and self.optimization_completed and self.rendering_performed):
            raise ReconstructionValidationError(
                "Phase 9.5C completed reports require real optimization and preview rendering."
            )
        if self.quality_evaluated or self.performance_qualified:
            raise ReconstructionValidationError(
                "Phase 9.5C cannot claim final NeRF quality or performance qualification."
            )
        if any(
            bool(value)
            for value in (
                self.canonical_geometry_authoritative,
                self.metric_scale_authoritative,
                self.measurement_authoritative,
                self.qc_authoritative,
                self.certified_metrology,
                self.artifact.canonical_geometry_authoritative,
                self.artifact.certified_metrology,
            )
        ):
            raise ReconstructionValidationError(
                "NeRF optimization cannot claim geometry/metric/measurement/QC/metrology authority."
            )
        object.__setattr__(self, "elapsed_seconds", elapsed)
        object.__setattr__(self, "real_backend_started", True)
        object.__setattr__(self, "optimization_completed", True)
        object.__setattr__(self, "rendering_performed", True)
        object.__setattr__(self, "quality_evaluated", False)
        object.__setattr__(self, "performance_qualified", False)
        object.__setattr__(self, "canonical_geometry_authoritative", False)
        object.__setattr__(self, "metric_scale_authoritative", False)
        object.__setattr__(self, "measurement_authoritative", False)
        object.__setattr__(self, "qc_authoritative", False)
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "schema_version", PHASE9_5C_NERF_OPTIMIZATION_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state": self.state.value,
            "request_id": self.request_id,
            "runtime": self.runtime_report.as_dict(),
            "artifact": self.artifact.as_dict(),
            "runtime_evidence_path": self.runtime_evidence_path,
            "preview_path": self.preview_path,
            "worker_console_log_path": self.worker_console_log_path,
            "optimization_manifest_path": self.optimization_manifest_path,
            "elapsed_seconds": self.elapsed_seconds,
            "real_backend_started": self.real_backend_started,
            "optimization_completed": self.optimization_completed,
            "rendering_performed": self.rendering_performed,
            "quality_evaluated": self.quality_evaluated,
            "performance_qualified": self.performance_qualified,
            "nerf_scene_role": self.artifact.scene_role.value,
            "canonical_geometry_authoritative": self.canonical_geometry_authoritative,
            "metric_scale_authoritative": self.metric_scale_authoritative,
            "measurement_authoritative": self.measurement_authoritative,
            "qc_authoritative": self.qc_authoritative,
            "certified_metrology": self.certified_metrology,
        }


def run_supervised_phase9_nerf_optimization(
    *,
    runtime_report: NerfInteractiveRuntimeReport,
    dataset: NeuralSceneDataset,
    dataset_manifest_path: str | Path,
    output_dir: str | Path,
    worker_script: str | Path,
    request_id: str,
    profile: NerfOptimizationProfile = NerfOptimizationProfile.EFFICIENT,
    training_steps: int = PHASE9_5C_DEFAULT_TRAINING_STEPS,
    maximum_input_frames: int = PHASE9_5C_DEFAULT_MAX_INPUT_FRAMES,
) -> NerfOptimizationExecutionReport:
    """Run the proven Phase 8 Nerfstudio worker under the Phase 9.5A lifecycle policy."""

    if not isinstance(runtime_report, NerfInteractiveRuntimeReport):
        raise TypeError("runtime_report must be NerfInteractiveRuntimeReport.")
    if runtime_report.state is not NerfInteractiveRuntimeState.READY:
        raise NerfWorkerExecutionError(
            "Phase 9.5C refuses optimization because the isolated NeRF runtime is not READY."
        )
    if runtime_report.spec.backend_kind is not NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL:
        raise NerfWorkerExecutionError("Phase 9.5C production optimization requires Nerfstudio/Nerfacto.")
    if current_process_nerf_heavy_modules():
        raise NerfWorkerExecutionError("Heavy NeRF modules leaked into the production depth interpreter.")
    if not isinstance(dataset, NeuralSceneDataset):
        raise TypeError("dataset must be NeuralSceneDataset.")
    manifest = Path(dataset_manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise NerfWorkerExecutionError(f"NeRF dataset manifest is missing: {manifest}")
    persisted = read_neural_scene_dataset_manifest(manifest)
    if neural_scene_dataset_fingerprint(persisted) != neural_scene_dataset_fingerprint(dataset):
        raise NerfWorkerExecutionError("NeRF input manifest does not match the supplied dataset evidence.")
    maximum = int(maximum_input_frames)
    if maximum < 1 or dataset.frame_count > maximum:
        raise NerfWorkerExecutionError("NeRF input frame count exceeds the configured bound.")
    steps = int(training_steps)
    if not 2 <= steps <= PHASE9_5C_MAX_TRAINING_STEPS:
        raise NerfWorkerExecutionError(
            f"Phase 9.5C training_steps must lie in [2, {PHASE9_5C_MAX_TRAINING_STEPS}]."
        )
    worker = Path(worker_script).expanduser().resolve()
    if not worker.is_file():
        raise NerfWorkerExecutionError(f"NeRF worker script is missing: {worker}")

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    request = NerfWorkerRequest.from_dataset(
        request_id=request_id,
        backend_kind=NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL,
        profile=profile,
        dataset=dataset,
        dataset_manifest_path=manifest,
        output_dir=(root / "worker-output").resolve(),
        maximum_input_frames=maximum,
        qualification_training_steps=steps,
    )

    policy = runtime_report.spec.process_policy
    console_log = (root / "nerfstudio_worker_console.log").resolve()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="phase9-5c-request-", dir=str(root)) as tmp:
        tmp_dir = Path(tmp)
        request_path = tmp_dir / "request.json"
        result_path = tmp_dir / "result.json"
        request_path.write_text(
            json.dumps(request.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        command = [
            runtime_report.spec.python_executable,
            str(worker),
            "--request",
            str(request_path),
            "--result",
            str(result_path),
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=str(worker.parent),
                env=_isolated_worker_environment(runtime_report),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=policy.start_new_session,
                shell=policy.shell,
            )
        except OSError as exc:
            raise NerfWorkerExecutionError("Could not start isolated Nerfstudio worker.") from exc
        try:
            stdout, _ = process.communicate(timeout=policy.timeout_s)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process, grace_s=policy.terminate_grace_s)
            raise NerfWorkerExecutionError(
                "Isolated Nerfstudio worker timed out and its process group was terminated."
            ) from exc
        console_log.write_text(stdout or "", encoding="utf-8")
        if process.returncode != 0:
            detail = (stdout or "").strip() or "unknown Nerfstudio worker failure"
            raise NerfWorkerExecutionError(
                f"Isolated Nerfstudio worker exited with code {process.returncode}: {detail}"
            )
        if not result_path.is_file():
            raise NerfWorkerExecutionError("Isolated Nerfstudio worker did not produce result.json.")
        artifact = NerfSceneArtifact.from_payload(_load_json_object(result_path, label="Nerfstudio result.json"))

    elapsed = float(time.monotonic() - started)
    if artifact.request_id != request.request_id:
        raise NerfWorkerExecutionError("NeRF artifact request_id does not match the request.")
    if artifact.dataset_fingerprint_sha256 != request.dataset_fingerprint_sha256:
        raise NerfWorkerExecutionError("NeRF artifact dataset fingerprint does not match the request.")
    if artifact.backend_provenance != nerf_backend_provenance(NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL):
        raise NerfWorkerExecutionError("NeRF artifact backend provenance is not the qualified policy.")
    if artifact.training_state is not NerfTrainingState.TRAINED or not artifact.render_capable:
        raise NerfWorkerExecutionError("NeRF artifact does not prove trained/render-capable execution.")
    if artifact.training_steps != steps or artifact.input_frame_count != dataset.frame_count:
        raise NerfWorkerExecutionError("NeRF artifact training/input counts are inconsistent.")

    artifact_dir = Path(artifact.artifact_dir).resolve()
    runtime_evidence_path = artifact_dir / "e2_runtime_evidence.json"
    preview_path = artifact_dir / "preview.png"
    config_path = Path(artifact.config_path).resolve()
    checkpoint_path = Path(artifact.checkpoint_path).resolve()
    for path, label in (
        (runtime_evidence_path, "NeRF runtime evidence"),
        (preview_path, "NeRF preview"),
        (config_path, "Nerfstudio config"),
        (checkpoint_path, "Nerfstudio checkpoint"),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise NerfWorkerExecutionError(f"{label} is missing or empty: {path}")

    runtime_evidence = _load_json_object(runtime_evidence_path, label="e2_runtime_evidence.json")
    if runtime_evidence.get("nerfstudio_revision") != runtime_report.spec.expected_nerfstudio_revision:
        raise NerfWorkerExecutionError("Worker Nerfstudio revision differs from the qualified 9.5A revision.")
    if runtime_evidence.get("tiny_cuda_nn_revision") != runtime_report.spec.expected_tiny_cuda_nn_revision:
        raise NerfWorkerExecutionError("Worker tiny-cuda-nn revision differs from the qualified 9.5A revision.")
    if runtime_evidence.get("training_completed_without_oom") is not True:
        raise NerfWorkerExecutionError("Worker evidence does not prove OOM-free bounded training.")
    if str(runtime_evidence.get("preview_path", "")) != str(preview_path):
        raise NerfWorkerExecutionError("Worker runtime evidence preview path mismatch.")
    if not str(artifact.device).lower().startswith("cuda:"):
        raise NerfWorkerExecutionError("Real Phase 9.5C execution did not report a CUDA device.")
    if current_process_nerf_heavy_modules():
        raise NerfWorkerExecutionError("Heavy NeRF modules leaked into depth after worker execution.")

    manifest_payload = {
        "phase": "9.5C",
        "schema_version": PHASE9_5C_NERF_OPTIMIZATION_SCHEMA_VERSION,
        "request_id": request.request_id,
        "backend_kind": NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL.value,
        "method_name": artifact.method_name,
        "dataset_fingerprint_sha256": artifact.dataset_fingerprint_sha256,
        "dataset_manifest_path": str(manifest),
        "input_frame_count": artifact.input_frame_count,
        "training_steps": artifact.training_steps,
        "training_duration_s": artifact.training_duration_s,
        "render_capable": artifact.render_capable,
        "device": artifact.device,
        "artifact_dir": str(artifact_dir),
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "preview_path": str(preview_path),
        "preview_sha256": _sha256_file(preview_path),
        "runtime_evidence_path": str(runtime_evidence_path),
        "runtime_evidence_sha256": _sha256_file(runtime_evidence_path),
        "nerf_scene_role": "appearance_only",
        "canonical_geometry_authoritative": False,
        "metric_scale_authoritative": False,
        "measurement_authoritative": False,
        "qc_authoritative": False,
        "certified_metrology": False,
        "quality_evaluated": False,
        "performance_qualified": False,
        "persistence_deferred_to": "9.5D",
        "interactive_viewer_deferred_to": "9.5E",
        "resource_lifecycle_deferred_to": "9.5G",
    }
    optimization_manifest = (root / "phase9_5c_optimization_manifest.json").resolve()
    optimization_manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return NerfOptimizationExecutionReport(
        state=NerfOptimizationExecutionState.COMPLETED,
        request_id=request.request_id,
        runtime_report=runtime_report,
        artifact=artifact,
        runtime_evidence_path=str(runtime_evidence_path),
        preview_path=str(preview_path),
        worker_console_log_path=str(console_log),
        optimization_manifest_path=str(optimization_manifest),
        elapsed_seconds=elapsed,
    )


__all__ = [
    "PHASE9_5C_NERF_OPTIMIZATION_SCHEMA_VERSION",
    "PHASE9_5C_DEFAULT_TRAINING_STEPS",
    "PHASE9_5C_MAX_TRAINING_STEPS",
    "PHASE9_5C_DEFAULT_MAX_INPUT_FRAMES",
    "NerfOptimizationExecutionState",
    "NerfOptimizationExecutionReport",
    "run_supervised_phase9_nerf_optimization",
]
