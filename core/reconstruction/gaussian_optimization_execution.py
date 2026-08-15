"""Phase 9.4C supervised real Gaussian optimization execution.

This module connects the Phase 9.4A qualified isolated gsplat runtime and the
Phase 9.4B canonical-scene ``NeuralSceneDataset`` bridge to the already-proven
Phase 8 gsplat worker.  Gaussian output remains appearance-only: the canonical
scene keeps geometry/metric/measurement/QC authority and no Gaussian artifact
may claim certified metrology.

The production ``depth`` interpreter never imports gsplat.  The heavy worker is
launched by absolute external interpreter path in its own process group with a
bounded timeout and SIGTERM -> SIGKILL escalation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite, log
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from typing import Any, Mapping

import numpy as np

from core.contracts.validation import require_nonempty
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.gaussian_interactive_runtime import (
    GaussianInteractiveRuntimeReport,
    GaussianInteractiveRuntimeState,
    current_process_gaussian_heavy_modules,
)
from core.reconstruction.gaussian_splatting import (
    GaussianBackendKind,
    GaussianOptimizationProfile,
    GaussianOptimizationState,
    GaussianSceneArtifact,
    GaussianSceneRole,
    GaussianSeed,
    GaussianWorkerExecutionError,
    GaussianWorkerRequest,
    gaussian_backend_provenance,
    write_gaussian_seed_npz,
)
from core.reconstruction.neural_scene import (
    NeuralSceneDataset,
    neural_scene_dataset_fingerprint,
    read_neural_scene_dataset_manifest,
)

PHASE9_4C_GAUSSIAN_OPTIMIZATION_SCHEMA_VERSION = 1
PHASE9_4C_DEFAULT_MAX_SEED_GAUSSIANS = 2048
PHASE9_4C_DEFAULT_MAX_INPUT_FRAMES = 8


class GaussianOptimizationExecutionState(str, Enum):
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class GaussianCanonicalSeedEvidence:
    source_surface_path: str
    source_surface_sha256: str
    gaussian_count: int
    source_vertex_count: int
    source_face_count: int
    positions_are_canonical_mesh_vertices: bool = True

    def __post_init__(self) -> None:
        path = Path(require_nonempty(self.source_surface_path, field_name="source_surface_path"))
        if not path.is_absolute():
            raise ReconstructionValidationError("Gaussian canonical seed source path must be absolute.")
        digest = str(self.source_surface_sha256).strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ReconstructionValidationError("source_surface_sha256 must be a SHA-256 hex digest.")
        for name in ("gaussian_count", "source_vertex_count", "source_face_count"):
            value = int(getattr(self, name))
            if value < 1:
                raise ReconstructionValidationError(f"{name} must be >= 1.")
            object.__setattr__(self, name, value)
        if self.gaussian_count > self.source_vertex_count:
            raise ReconstructionValidationError("Gaussian seed cannot contain more entries than source vertices.")
        if not bool(self.positions_are_canonical_mesh_vertices):
            raise ReconstructionValidationError("Phase 9.4C seed positions must remain canonical mesh vertices.")
        object.__setattr__(self, "source_surface_path", str(path))
        object.__setattr__(self, "source_surface_sha256", digest)
        object.__setattr__(self, "positions_are_canonical_mesh_vertices", True)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_surface_path": self.source_surface_path,
            "source_surface_sha256": self.source_surface_sha256,
            "gaussian_count": self.gaussian_count,
            "source_vertex_count": self.source_vertex_count,
            "source_face_count": self.source_face_count,
            "positions_are_canonical_mesh_vertices": self.positions_are_canonical_mesh_vertices,
        }


def build_gaussian_seed_from_canonical_arrays(
    vertices_reconstruction: np.ndarray,
    faces: np.ndarray,
    vertex_colors_rgba: np.ndarray,
    *,
    max_gaussians: int = PHASE9_4C_DEFAULT_MAX_SEED_GAUSSIANS,
) -> GaussianSeed:
    """Create a bounded seed directly from an integrity-verified canonical mesh.

    No geometry is reconstructed or optimized here.  Positions are a deterministic
    subset of the exact canonical vertices and only color/opacity are trainable in
    the Phase 8 qualification worker reused by 9.4C.
    """

    vertices = np.asarray(vertices_reconstruction, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    colors = np.asarray(vertex_colors_rgba, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3:
        raise ReconstructionValidationError("Canonical Gaussian seed vertices must be finite Nx3.")
    if not np.isfinite(vertices).all():
        raise ReconstructionValidationError("Canonical Gaussian seed vertices must be finite.")
    if triangles.ndim != 2 or triangles.shape[1:] != (3,) or len(triangles) < 1:
        raise ReconstructionValidationError("Canonical Gaussian seed requires triangular faces.")
    if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise ReconstructionValidationError("Canonical Gaussian seed faces reference missing vertices.")
    if colors.shape != (len(vertices), 4) or not np.isfinite(colors).all():
        raise ReconstructionValidationError("Canonical Gaussian seed colors must be finite Nx4 RGBA.")
    if np.any(colors < 0.0) or np.any(colors > 1.0):
        raise ReconstructionValidationError("Canonical Gaussian seed colors must lie in [0, 1].")
    if isinstance(max_gaussians, bool) or int(max_gaussians) < 1:
        raise ReconstructionValidationError("max_gaussians must be >= 1.")
    maximum = int(max_gaussians)

    indices = np.arange(len(vertices), dtype=np.int64)
    if len(indices) > maximum:
        pick = np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)
        indices = indices[pick]

    sampled_faces = triangles
    if len(sampled_faces) > 20_000:
        pick = np.linspace(0, len(sampled_faces) - 1, 20_000, dtype=np.int64)
        sampled_faces = sampled_faces[pick]
    edges = np.concatenate(
        [sampled_faces[:, [0, 1]], sampled_faces[:, [1, 2]], sampled_faces[:, [2, 0]]], axis=0
    )
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    lengths = lengths[np.isfinite(lengths) & (lengths > 0.0)]
    if len(lengths) == 0:
        raise ReconstructionValidationError("Canonical surface has no usable positive edge length.")
    base_scale = max(float(np.median(lengths)) * 0.35, np.finfo(np.float32).tiny)
    log_scale = float(log(base_scale))

    rgb_u8 = np.rint(colors[indices, :3] * 255.0).clip(0, 255).astype(np.uint8)
    count = len(indices)
    return GaussianSeed(
        positions_reconstruction=vertices[indices].astype(np.float32),
        colors_rgb=rgb_u8,
        log_scales_reconstruction=np.full((count, 3), log_scale, dtype=np.float32),
        opacities=np.full((count,), 0.90, dtype=np.float32),
        quality_scores=np.ones((count,), dtype=np.float32),
        source_vertex_indices=indices.astype(np.uint32),
    )


def _isolated_worker_environment(python_executable: str) -> dict[str, str]:
    """Build a child-only conda/CUDA environment without mutating ``depth``."""

    python = Path(python_executable).resolve()
    prefix = python.parent.parent
    if python.parent.name != "bin":
        raise GaussianWorkerExecutionError(f"Cannot infer Gaussian conda prefix from {python}.")
    env = dict(os.environ)
    env["CONDA_PREFIX"] = str(prefix)
    env["CONDA_DEFAULT_ENV"] = prefix.name
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONPATH", None)

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
    lib_dirs = [prefix / "lib", prefix / "lib64", prefix / "targets" / "x86_64-linux" / "lib"]
    prepend("LD_LIBRARY_PATH", lib_dirs)
    prepend("LIBRARY_PATH", lib_dirs)
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
        # The OS should reap a SIGKILL'ed process; retain a bounded final wait.
        pass


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GaussianWorkerExecutionError(f"Could not parse {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GaussianWorkerExecutionError(f"{label} must be a JSON object.")
    return payload


@dataclass(frozen=True, slots=True)
class GaussianOptimizationExecutionReport:
    state: GaussianOptimizationExecutionState
    request_id: str
    runtime_report: GaussianInteractiveRuntimeReport
    artifact: GaussianSceneArtifact
    seed_evidence: GaussianCanonicalSeedEvidence
    qualification_metrics_path: str
    preview_path: str
    worker_console_log_path: str
    elapsed_seconds: float
    real_backend_started: bool = True
    optimization_completed: bool = True
    rendering_performed: bool = True
    canonical_geometry_authoritative: bool = False
    metric_scale_authoritative: bool = False
    measurement_authoritative: bool = False
    qc_authoritative: bool = False
    certified_metrology: bool = False
    schema_version: int = PHASE9_4C_GAUSSIAN_OPTIMIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.state is not GaussianOptimizationExecutionState.COMPLETED:
            raise ReconstructionValidationError("Successful Phase 9.4C reports must be COMPLETED.")
        object.__setattr__(self, "request_id", require_nonempty(self.request_id, field_name="request_id"))
        if self.runtime_report.state is not GaussianInteractiveRuntimeState.READY:
            raise ReconstructionValidationError("Phase 9.4C execution requires a READY 9.4A runtime report.")
        if self.artifact.request_id != self.request_id:
            raise ReconstructionValidationError("Gaussian execution artifact request_id mismatch.")
        if self.artifact.optimization_state is not GaussianOptimizationState.OPTIMIZED:
            raise ReconstructionValidationError("Phase 9.4C requires an optimized Gaussian artifact.")
        if self.artifact.scene_role is not GaussianSceneRole.APPEARANCE_ONLY:
            raise ReconstructionValidationError("Gaussian output must remain appearance-only.")
        if self.artifact.backend_provenance != gaussian_backend_provenance(GaussianBackendKind.GSPLAT_EXTERNAL):
            raise ReconstructionValidationError("Phase 9.4C requires the permissive gsplat backend.")
        elapsed = float(self.elapsed_seconds)
        if not isfinite(elapsed) or elapsed <= 0.0:
            raise ReconstructionValidationError("elapsed_seconds must be finite and positive.")
        for name in ("qualification_metrics_path", "preview_path", "worker_console_log_path"):
            path = Path(require_nonempty(getattr(self, name), field_name=name))
            if not path.is_absolute():
                raise ReconstructionValidationError(f"{name} must be absolute.")
            object.__setattr__(self, name, str(path))
        if not bool(self.real_backend_started) or not bool(self.optimization_completed) or not bool(self.rendering_performed):
            raise ReconstructionValidationError("Phase 9.4C completed reports require real optimization and preview rendering.")
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
            raise ReconstructionValidationError("Gaussian optimization cannot claim geometry/metric/measurement/QC/metrology authority.")
        object.__setattr__(self, "elapsed_seconds", elapsed)
        object.__setattr__(self, "real_backend_started", True)
        object.__setattr__(self, "optimization_completed", True)
        object.__setattr__(self, "rendering_performed", True)
        object.__setattr__(self, "canonical_geometry_authoritative", False)
        object.__setattr__(self, "metric_scale_authoritative", False)
        object.__setattr__(self, "measurement_authoritative", False)
        object.__setattr__(self, "qc_authoritative", False)
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "schema_version", PHASE9_4C_GAUSSIAN_OPTIMIZATION_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state": self.state.value,
            "request_id": self.request_id,
            "runtime": self.runtime_report.as_dict(),
            "artifact": self.artifact.as_dict(),
            "seed_evidence": self.seed_evidence.as_dict(),
            "qualification_metrics_path": self.qualification_metrics_path,
            "preview_path": self.preview_path,
            "worker_console_log_path": self.worker_console_log_path,
            "elapsed_seconds": self.elapsed_seconds,
            "real_backend_started": self.real_backend_started,
            "optimization_completed": self.optimization_completed,
            "rendering_performed": self.rendering_performed,
            "gaussian_scene_role": self.artifact.scene_role.value,
            "canonical_geometry_authoritative": self.canonical_geometry_authoritative,
            "metric_scale_authoritative": self.metric_scale_authoritative,
            "measurement_authoritative": self.measurement_authoritative,
            "qc_authoritative": self.qc_authoritative,
            "certified_metrology": self.certified_metrology,
        }


def run_supervised_phase9_gaussian_optimization(
    *,
    runtime_report: GaussianInteractiveRuntimeReport,
    dataset: NeuralSceneDataset,
    dataset_manifest_path: str | Path,
    seed: GaussianSeed,
    seed_evidence: GaussianCanonicalSeedEvidence,
    output_dir: str | Path,
    worker_script: str | Path,
    request_id: str,
    profile: GaussianOptimizationProfile = GaussianOptimizationProfile.EFFICIENT,
    maximum_input_frames: int = PHASE9_4C_DEFAULT_MAX_INPUT_FRAMES,
) -> GaussianOptimizationExecutionReport:
    """Run the existing real gsplat worker under the Phase 9.4A lifecycle policy."""

    if not isinstance(runtime_report, GaussianInteractiveRuntimeReport):
        raise TypeError("runtime_report must be GaussianInteractiveRuntimeReport.")
    if runtime_report.state is not GaussianInteractiveRuntimeState.READY:
        raise GaussianWorkerExecutionError(
            "Phase 9.4C refuses optimization because the isolated gsplat runtime is not READY."
        )
    if runtime_report.spec.backend_kind is not GaussianBackendKind.GSPLAT_EXTERNAL:
        raise GaussianWorkerExecutionError("Phase 9.4C production optimization requires gsplat_external.")
    if current_process_gaussian_heavy_modules():
        raise GaussianWorkerExecutionError("Heavy Gaussian modules leaked into the production depth interpreter.")
    if not isinstance(dataset, NeuralSceneDataset):
        raise TypeError("dataset must be NeuralSceneDataset.")
    manifest = Path(dataset_manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise GaussianWorkerExecutionError(f"Gaussian dataset manifest is missing: {manifest}")
    persisted_dataset = read_neural_scene_dataset_manifest(manifest)
    if neural_scene_dataset_fingerprint(persisted_dataset) != neural_scene_dataset_fingerprint(dataset):
        raise GaussianWorkerExecutionError("Gaussian input manifest does not match the supplied dataset evidence.")
    if dataset.frame_count > int(maximum_input_frames):
        raise GaussianWorkerExecutionError("Gaussian input frame count exceeds the configured bound.")
    worker = Path(worker_script).expanduser().resolve()
    if not worker.is_file():
        raise GaussianWorkerExecutionError(f"Gaussian worker script is missing: {worker}")

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    seed_path = write_gaussian_seed_npz(seed, (root / "gaussian_seed.npz").resolve())
    request = GaussianWorkerRequest.from_dataset(
        request_id=request_id,
        backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL,
        profile=profile,
        dataset=dataset,
        dataset_manifest_path=manifest,
        seed_path=seed_path,
        output_dir=(root / "worker-output").resolve(),
        maximum_input_frames=maximum_input_frames,
    )

    policy = runtime_report.spec.process_policy
    console_log = (root / "gsplat_worker_console.log").resolve()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="phase9-4c-request-", dir=str(root)) as tmp:
        tmp_dir = Path(tmp)
        request_path = tmp_dir / "request.json"
        result_path = tmp_dir / "result.json"
        request_path.write_text(json.dumps(request.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
                env=_isolated_worker_environment(runtime_report.spec.python_executable),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=policy.start_new_session,
                shell=policy.shell,
            )
        except OSError as exc:
            raise GaussianWorkerExecutionError("Could not start isolated gsplat worker.") from exc
        try:
            stdout, _ = process.communicate(timeout=policy.timeout_s)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process, grace_s=policy.terminate_grace_s)
            raise GaussianWorkerExecutionError("Isolated gsplat worker timed out and was terminated.") from exc
        console_log.write_text(stdout or "", encoding="utf-8")
        if process.returncode != 0:
            detail = (stdout or "").strip() or "unknown gsplat worker failure"
            raise GaussianWorkerExecutionError(
                f"Isolated gsplat worker exited with code {process.returncode}: {detail}"
            )
        if not result_path.is_file():
            raise GaussianWorkerExecutionError("Isolated gsplat worker did not produce result.json.")
        payload = _load_json_object(result_path, label="gsplat result.json")
        artifact = GaussianSceneArtifact.from_payload(payload)

    elapsed = float(time.monotonic() - started)
    if artifact.request_id != request.request_id:
        raise GaussianWorkerExecutionError("Gaussian artifact request_id does not match the request.")
    if artifact.dataset_fingerprint_sha256 != request.dataset_fingerprint_sha256:
        raise GaussianWorkerExecutionError("Gaussian artifact dataset fingerprint does not match the request.")
    if artifact.backend_provenance != gaussian_backend_provenance(GaussianBackendKind.GSPLAT_EXTERNAL):
        raise GaussianWorkerExecutionError("Gaussian artifact backend provenance is not the qualified gsplat policy.")
    scene_path = Path(artifact.scene_path)
    artifact_dir = Path(artifact.artifact_dir)
    metrics_path = artifact_dir / "qualification_metrics.json"
    preview_path = artifact_dir / "preview.png"
    artifact_manifest = artifact_dir / "artifact_manifest.json"
    for path, label in (
        (scene_path, "optimized Gaussian scene"),
        (metrics_path, "Gaussian qualification metrics"),
        (preview_path, "Gaussian preview"),
        (artifact_manifest, "Gaussian artifact manifest"),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise GaussianWorkerExecutionError(f"{label} is missing or empty: {path}")
    metrics = _load_json_object(metrics_path, label="qualification_metrics.json")
    if metrics.get("backend_kind") != GaussianBackendKind.GSPLAT_EXTERNAL.value:
        raise GaussianWorkerExecutionError("Gaussian metrics backend kind mismatch.")
    if metrics.get("real_cuda_execution") is not True or metrics.get("optimization_completed") is not True:
        raise GaussianWorkerExecutionError("Gaussian metrics do not prove real completed CUDA optimization.")
    if metrics.get("render_written") is not True:
        raise GaussianWorkerExecutionError("Gaussian optimization did not produce a preview render.")
    peak = float(metrics.get("peak_vram_mib", 0.0) or 0.0)
    if not isfinite(peak) or peak <= 0.0 or (artifact.peak_vram_mib or 0.0) <= 0.0:
        raise GaussianWorkerExecutionError("Real Gaussian optimization requires positive measured CUDA VRAM.")
    if artifact.training_steps < 1 or artifact.gaussian_count != seed.gaussian_count:
        raise GaussianWorkerExecutionError("Gaussian artifact training/seed counts are inconsistent.")
    if current_process_gaussian_heavy_modules():
        raise GaussianWorkerExecutionError("Heavy Gaussian modules leaked into depth after worker execution.")

    return GaussianOptimizationExecutionReport(
        state=GaussianOptimizationExecutionState.COMPLETED,
        request_id=request.request_id,
        runtime_report=runtime_report,
        artifact=artifact,
        seed_evidence=seed_evidence,
        qualification_metrics_path=str(metrics_path.resolve()),
        preview_path=str(preview_path.resolve()),
        worker_console_log_path=str(console_log),
        elapsed_seconds=elapsed,
    )


__all__ = [
    "PHASE9_4C_GAUSSIAN_OPTIMIZATION_SCHEMA_VERSION",
    "PHASE9_4C_DEFAULT_MAX_SEED_GAUSSIANS",
    "PHASE9_4C_DEFAULT_MAX_INPUT_FRAMES",
    "GaussianOptimizationExecutionState",
    "GaussianCanonicalSeedEvidence",
    "GaussianOptimizationExecutionReport",
    "build_gaussian_seed_from_canonical_arrays",
    "run_supervised_phase9_gaussian_optimization",
]
