"""F1 orchestration binding for the isolated Depth Anything 3 provider."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import isfinite
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from core.contracts.validation import JsonValue, metadata_as_dict, normalize_metadata, require_nonempty
from core.reconstruction.da3 import (
    DA3_ADAPTER_VERSION,
    DA3_ALLOWED_MODEL_IDS,
    DA3_EFFICIENT_MODEL_ID,
    DA3_GEOMETRY_SCALE,
    DA3_PROVIDER_ID,
    DA3_SCHEMA_VERSION,
    DA3EnvironmentProbe,
    DA3ResultManifest,
    DA3WorkerExecutionError,
    read_da3_result_manifest,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.orchestration import (
    ReconstructionArtifactAuthority,
    ReconstructionArtifactKind,
    ReconstructionPipelineArtifact,
    ReconstructionPipelineStage,
    ReconstructionProviderDescriptor,
    ReconstructionProviderExecution,
    ReconstructionProviderRole,
    ReconstructionStageExecutionContext,
)


@dataclass(frozen=True, slots=True)
class DA3ProviderProfile:
    model_id: str = DA3_EFFICIENT_MODEL_ID
    model_revision: str = ""
    process_res: int = 392
    process_res_method: str = "upper_bound_resize"
    use_ray_pose: bool = False
    ref_view_strategy: str = "saddle_balanced"

    def __post_init__(self) -> None:
        model = require_nonempty(self.model_id, field_name="model_id")
        if model not in DA3_ALLOWED_MODEL_IDS:
            raise ReconstructionValidationError(f"Unsupported F2 DA3 model_id: {model!r}.")
        revision = str(self.model_revision).strip().lower()
        if revision and (len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision)):
            raise ReconstructionValidationError("DA3 model_revision must be empty or a full 40-character hex revision.")
        resolution = int(self.process_res)
        if resolution < 196 or resolution > 1008:
            raise ReconstructionValidationError("DA3 process_res must be within [196, 1008].")
        method = str(self.process_res_method).strip()
        if method not in {"upper_bound_resize", "lower_bound_resize"}:
            raise ReconstructionValidationError("Unsupported DA3 process_res_method.")
        strategy = str(self.ref_view_strategy).strip()
        if strategy not in {"first", "middle", "saddle_balanced", "saddle_sim_range"}:
            raise ReconstructionValidationError("Unsupported DA3 ref_view_strategy.")
        object.__setattr__(self, "model_id", model)
        object.__setattr__(self, "model_revision", revision)
        object.__setattr__(self, "process_res", resolution)
        object.__setattr__(self, "process_res_method", method)
        object.__setattr__(self, "use_ray_pose", bool(self.use_ray_pose))
        object.__setattr__(self, "ref_view_strategy", strategy)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "process_res": self.process_res,
            "process_res_method": self.process_res_method,
            "use_ray_pose": self.use_ray_pose,
            "ref_view_strategy": self.ref_view_strategy,
        }


@dataclass(frozen=True, slots=True)
class DA3InputManifest:
    request_id: str
    image_paths: tuple[str, ...]
    metadata: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(default_factory=tuple)
    schema_version: int = DA3_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", require_nonempty(self.request_id, field_name="request_id"))
        paths = tuple(str(Path(value).absolute()) for value in self.image_paths)
        if not paths:
            raise ReconstructionValidationError("DA3 input manifest requires at least one image.")
        if len(set(paths)) != len(paths):
            raise ReconstructionValidationError("DA3 input image paths must be unique.")
        if any(not Path(value).is_absolute() for value in paths):
            raise ReconstructionValidationError("DA3 input image paths must be absolute.")
        object.__setattr__(self, "image_paths", paths)
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))
        if int(self.schema_version) != DA3_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported DA3 input schema version.")
        object.__setattr__(self, "schema_version", DA3_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "image_paths": list(self.image_paths),
            "metadata": metadata_as_dict(self.metadata),  # type: ignore[arg-type]
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DA3InputManifest":
        if not isinstance(payload, Mapping):
            raise ReconstructionValidationError("DA3 input manifest must be a JSON object.")
        try:
            return cls(
                request_id=str(payload["request_id"]),
                image_paths=tuple(str(v) for v in payload["image_paths"]),
                metadata=payload.get("metadata", {}),
                schema_version=int(payload.get("schema_version", DA3_SCHEMA_VERSION)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReconstructionValidationError(f"Invalid DA3 input manifest: {exc}") from exc


def write_da3_input_manifest(path: str | Path, manifest: DA3InputManifest) -> Path:
    if not isinstance(manifest, DA3InputManifest):
        raise TypeError("manifest must be DA3InputManifest.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def read_da3_input_manifest(path: str | Path) -> DA3InputManifest:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read DA3 input manifest: {exc}") from exc
    return DA3InputManifest.from_payload(payload)


def da3_provider_descriptor(
    probe: DA3EnvironmentProbe,
    *,
    priority: int = 250,
) -> ReconstructionProviderDescriptor:
    if not isinstance(probe, DA3EnvironmentProbe):
        raise TypeError("probe must be DA3EnvironmentProbe.")
    return ReconstructionProviderDescriptor(
        provider_id=DA3_PROVIDER_ID,
        provider_version=probe.source_revision[:12] if probe.source_revision else "unqualified",
        adapter_version=DA3_ADAPTER_VERSION,
        roles=(ReconstructionProviderRole.CAMERA_POSE, ReconstructionProviderRole.DEPTH),
        execution=ReconstructionProviderExecution.EXTERNAL_PROCESS,
        priority=priority,
        available=probe.available,
        unavailable_reason="" if probe.available else (probe.error or "da3_environment_not_qualified"),
        metadata={
            "geometry_scale": DA3_GEOMETRY_SCALE,
            "pose_output": True,
            "depth_output": True,
            "provider_default_priority": priority,
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        },
    )


class DA3ExternalWorkerRunner:
    def __init__(
        self,
        *,
        python_executable: str | Path,
        worker_script: str | Path,
        profile: DA3ProviderProfile | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        self.python_executable = Path(python_executable)
        self.worker_script = Path(worker_script)
        self.profile = profile or DA3ProviderProfile()
        self.timeout_s = float(timeout_s)
        if not self.python_executable.is_absolute() or not self.worker_script.is_absolute():
            raise ReconstructionValidationError("DA3 worker paths must be absolute.")
        if not isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ReconstructionValidationError("DA3 worker timeout_s must be finite and positive.")

    def run(self, input_manifest_path: str | Path, output_dir: str | Path) -> tuple[Path, DA3ResultManifest]:
        source = Path(input_manifest_path)
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        request_path = destination / "worker_request.json"
        request_payload = {
            "schema_version": DA3_SCHEMA_VERSION,
            "input_manifest_path": str(source.absolute()),
            "output_dir": str(destination.absolute()),
            "profile": self.profile.as_dict(),
        }
        request_path.write_text(json.dumps(request_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            completed = subprocess.run(
                [str(self.python_executable), str(self.worker_script), "--request", str(request_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DA3WorkerExecutionError(f"DA3 worker launch failed: {type(exc).__name__}: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or f"exit_{completed.returncode}"
            raise DA3WorkerExecutionError(f"DA3 worker failed: {detail}")
        manifest_path = destination / "result_manifest.json"
        if not manifest_path.is_file():
            raise DA3WorkerExecutionError("DA3 worker did not produce result_manifest.json.")
        return manifest_path, read_da3_result_manifest(manifest_path)


class DepthAnything3ProviderExecutor:
    """One F1 executor exposing DA3 as camera-pose + depth supporting evidence.

    A single DA3 inference is cached per orchestration request and reused across the
    CAMERA_POSE and DEPTH_EVIDENCE stages.  DA3 is intentionally given a conservative
    default priority so it does not silently replace the qualified SfM/SLAM authority.
    """

    def __init__(self, *, runner: Any, workspace_root: str | Path | None = None) -> None:
        if not hasattr(runner, "run"):
            raise TypeError("runner must expose run(input_manifest_path, output_dir).")
        self.runner = runner
        self.workspace_root = Path(workspace_root or Path(tempfile.gettempdir()) / "eia-da3-provider")
        self._cache: dict[tuple[str, str], tuple[Path, DA3ResultManifest]] = {}

    @staticmethod
    def _input_manifest_path(context: ReconstructionStageExecutionContext) -> Path:
        ingestion = next(
            (artifact for artifact in context.required_inputs + context.optional_inputs if artifact.kind is ReconstructionArtifactKind.INGESTION_RESULT),
            None,
        )
        if ingestion is None:
            # DEPTH stage receives ingestion directly; CAMERA stage also receives it.
            raise ReconstructionValidationError("DA3 provider requires ingestion_result input.")
        ref = str(ingestion.payload_ref).strip()
        if ref.startswith("file://"):
            ref = ref[7:]
        path = Path(ref)
        if not path.is_absolute():
            raise ReconstructionValidationError("DA3 ingestion payload_ref must reference an absolute input manifest.")
        return path

    def _result(self, context: ReconstructionStageExecutionContext) -> tuple[Path, DA3ResultManifest]:
        key = (context.request.session_id, context.request.request_id)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        source = self._input_manifest_path(context)
        input_manifest = read_da3_input_manifest(source)
        if input_manifest.request_id != context.request.request_id:
            raise ReconstructionValidationError("DA3 input request_id does not match orchestration request.")
        output_dir = self.workspace_root / context.request.session_id / context.request.request_id
        result = self.runner.run(source, output_dir)
        self._cache[key] = result
        return result

    def execute(self, context: ReconstructionStageExecutionContext) -> ReconstructionPipelineArtifact:
        if not isinstance(context, ReconstructionStageExecutionContext):
            raise TypeError("context must be ReconstructionStageExecutionContext.")
        if context.provider.provider_id != DA3_PROVIDER_ID:
            raise ReconstructionValidationError("DA3 executor/provider identity mismatch.")
        if context.planned_stage.stage not in {ReconstructionPipelineStage.CAMERA_POSE, ReconstructionPipelineStage.DEPTH_EVIDENCE}:
            raise ReconstructionValidationError("DA3 provider only binds camera_pose and depth_evidence in F2.")
        manifest_path, result = self._result(context)
        if context.planned_stage.stage is ReconstructionPipelineStage.CAMERA_POSE:
            kind = ReconstructionArtifactKind.CAMERA_TRAJECTORY
            authority = ReconstructionArtifactAuthority.SUPPORTING_GEOMETRY
        else:
            kind = ReconstructionArtifactKind.DEPTH_EVIDENCE
            authority = ReconstructionArtifactAuthority.SUPPORTING_GEOMETRY
        return ReconstructionPipelineArtifact(
            artifact_id=f"{context.request.request_id}:{context.planned_stage.stage.value}:{DA3_PROVIDER_ID}",
            session_id=context.request.session_id,
            stage=context.planned_stage.stage,
            kind=kind,
            authority=authority,
            provider_id=DA3_PROVIDER_ID,
            payload_ref=str(manifest_path.absolute()),
            metadata={
                "model_id": result.model_id,
                "model_revision": result.model_revision,
                "source_revision": result.source_revision,
                "frame_count": result.frame_count,
                "geometry_scale": result.geometry_scale,
                "camera_axes": result.camera_axes,
                "extrinsics_convention": result.extrinsics_convention,
                "canonical_geometry_authoritative": False,
                "certified_metrology": False,
                "single_inference_reused_across_pose_and_depth": True,
            },
        )
