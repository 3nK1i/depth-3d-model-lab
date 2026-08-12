"""Authority-aware product export contracts for Phase 8.9A1.

This module productizes already-qualified reconstruction artifact writers without
changing reconstruction, persistence, checkpoint, recovery, or backend
lifecycles.  Sparse SfM remains supporting/arbitrary-scale geometry; only the
canonical SURFACE_MESH pipeline artifact may produce the canonical mesh export;
and inspection/QC remains inspection-decision evidence.  No export path may
claim certified metrology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, TypeAlias

from core.contracts.validation import require_nonempty
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.inspection_qc import InspectionReport, write_inspection_report_json
from core.reconstruction.orchestration import (
    ReconstructionArtifactAuthority,
    ReconstructionArtifactKind,
    ReconstructionPipelineArtifact,
)
from core.reconstruction.surface_mesh import SurfaceMeshResult, write_surface_mesh_obj

RECONSTRUCTION_EXPORT_SCHEMA = "eia.reconstruction.export.v1"
RECONSTRUCTION_EXPORT_MANIFEST = "reconstruction_export_manifest.json"
RECONSTRUCTION_EXPORT_CHECKSUM = "reconstruction_export_manifest.sha256"


@dataclass(frozen=True, slots=True)
class OfflineSparseExportSource:
    """Validated Phase 8.8 offline-worker sparse SfM output.

    The source is intentionally the dependency-neutral worker result mapping
    already consumed by the application.  A1 copies its complete sparse PLY and
    SfM JSON rather than importing PyCOLMAP or rebuilding sparse geometry in the
    production environment.
    """

    result_payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        payload = self.result_payload
        if not isinstance(payload, Mapping):
            raise TypeError("result_payload must be a mapping.")
        if str(payload.get("status", "")).strip().lower() != "completed":
            raise ReconstructionValidationError(
                "Sparse export requires a completed offline reconstruction result."
            )
        if not bool(payload.get("pass", False)):
            raise ReconstructionValidationError(
                "Sparse export requires an offline reconstruction that passed its quality gate."
            )
        if not bool(payload.get("camera_pose_available", False)):
            raise ReconstructionValidationError(
                "Sparse export requires available camera-pose evidence."
            )
        if bool(payload.get("canonical_geometry_authoritative", False)):
            raise ReconstructionValidationError(
                "Sparse SfM must not be promoted to canonical geometry authority."
            )
        if bool(payload.get("certified_metrology", False)):
            raise ReconstructionValidationError(
                "Sparse SfM export must not claim certified metrology."
            )
        session_id = require_nonempty(str(payload.get("session_id", "")), field_name="session_id")
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ReconstructionValidationError(
                "Completed offline reconstruction result is missing its artifacts mapping."
            )
        sparse_path = _required_absolute_file(
            artifacts.get("sparse_reconstruction_ply"),
            field_name="sparse_reconstruction_ply",
        )
        sfm_path = _required_absolute_file(
            artifacts.get("sfm_result_json"),
            field_name="sfm_result_json",
        )
        if sparse_path.suffix.lower() != ".ply":
            raise ReconstructionValidationError("Sparse reconstruction artifact must be a PLY file.")
        if sfm_path.suffix.lower() != ".json":
            raise ReconstructionValidationError("SfM result artifact must be a JSON file.")
        sfm_payload = _read_json_mapping(sfm_path, field_name="sfm_result_json")
        if str(sfm_payload.get("session_id", "")).strip() != session_id:
            raise ReconstructionValidationError(
                "Offline result and SfM artifact session identities differ."
            )
        gauge = sfm_payload.get("gauge")
        if not isinstance(gauge, Mapping):
            raise ReconstructionValidationError("SfM result artifact is missing gauge evidence.")
        if str(gauge.get("scale_status", "")).strip() != "arbitrary":
            raise ReconstructionValidationError(
                "Phase 8.9A1 sparse export only accepts the qualified arbitrary SfM gauge."
            )
        if gauge.get("unit_to_metre") is not None:
            raise ReconstructionValidationError(
                "Arbitrary sparse SfM must not expose unit_to_metre."
            )

    @property
    def session_id(self) -> str:
        return str(self.result_payload["session_id"])

    @property
    def artifacts(self) -> Mapping[str, Any]:
        value = self.result_payload["artifacts"]
        assert isinstance(value, Mapping)
        return value


@dataclass(frozen=True, slots=True)
class CanonicalSurfaceMeshExportSource:
    """Canonical mesh payload bound to its authoritative pipeline artifact."""

    artifact: ReconstructionPipelineArtifact
    result: SurfaceMeshResult
    canonical_geometry_artifact_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ReconstructionPipelineArtifact):
            raise TypeError("artifact must be ReconstructionPipelineArtifact.")
        if not isinstance(self.result, SurfaceMeshResult):
            raise TypeError("result must be SurfaceMeshResult.")
        canonical_id = require_nonempty(
            self.canonical_geometry_artifact_id,
            field_name="canonical_geometry_artifact_id",
        )
        if self.artifact.kind is not ReconstructionArtifactKind.SURFACE_MESH:
            raise ReconstructionValidationError(
                "Canonical mesh export requires a SURFACE_MESH artifact."
            )
        if self.artifact.authority is not ReconstructionArtifactAuthority.CANONICAL_GEOMETRY:
            raise ReconstructionValidationError(
                "Canonical mesh export requires canonical-geometry authority."
            )
        if self.artifact.artifact_id != canonical_id:
            raise ReconstructionValidationError(
                "Canonical mesh export requires the pipeline's current canonical geometry artifact."
            )
        if self.artifact.session_id != self.result.session_id:
            raise ReconstructionValidationError(
                "Surface artifact and surface payload belong to different sessions."
            )
        if bool(self.result.certified_metrology):
            raise ReconstructionValidationError(
                "Canonical surface export must not claim certified metrology."
            )
        object.__setattr__(self, "canonical_geometry_artifact_id", canonical_id)

    @property
    def session_id(self) -> str:
        return self.artifact.session_id


@dataclass(frozen=True, slots=True)
class InspectionReportExportSource:
    """Inspection report payload bound to inspection-decision authority."""

    artifact: ReconstructionPipelineArtifact
    report: InspectionReport

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ReconstructionPipelineArtifact):
            raise TypeError("artifact must be ReconstructionPipelineArtifact.")
        if not isinstance(self.report, InspectionReport):
            raise TypeError("report must be InspectionReport.")
        if self.artifact.kind is not ReconstructionArtifactKind.INSPECTION_REPORT:
            raise ReconstructionValidationError(
                "QC export requires an INSPECTION_REPORT artifact."
            )
        if self.artifact.authority is not ReconstructionArtifactAuthority.INSPECTION_DECISION:
            raise ReconstructionValidationError(
                "QC export requires inspection-decision authority."
            )
        if self.artifact.session_id != self.report.session_id:
            raise ReconstructionValidationError(
                "Inspection artifact and report belong to different sessions."
            )
        if bool(self.report.certified_metrology):
            raise ReconstructionValidationError(
                "Inspection report export must not claim certified metrology."
            )

    @property
    def session_id(self) -> str:
        return self.artifact.session_id


ReconstructionExportSource: TypeAlias = (
    OfflineSparseExportSource | CanonicalSurfaceMeshExportSource | InspectionReportExportSource
)


@dataclass(frozen=True, slots=True)
class ReconstructionExportRequest:
    output_dir: Path
    sources: tuple[ReconstructionExportSource, ...]

    def __post_init__(self) -> None:
        target = Path(self.output_dir).expanduser()
        if not target.is_absolute():
            raise ReconstructionValidationError("Reconstruction export output_dir must be absolute.")
        target = target.resolve()
        sources = tuple(self.sources)
        if not sources:
            raise ReconstructionValidationError("Reconstruction export requires at least one source.")
        allowed = (OfflineSparseExportSource, CanonicalSurfaceMeshExportSource, InspectionReportExportSource)
        if any(not isinstance(source, allowed) for source in sources):
            raise TypeError("sources contain an unsupported reconstruction export source.")
        if len({type(source) for source in sources}) != len(sources):
            raise ReconstructionValidationError(
                "A reconstruction export may contain each source kind at most once."
            )
        session_ids = {source.session_id for source in sources}
        if len(session_ids) != 1:
            raise ReconstructionValidationError(
                "All reconstruction export sources must belong to the same session."
            )
        if target.exists():
            raise ReconstructionValidationError(
                f"Reconstruction export destination already exists: {target}"
            )
        object.__setattr__(self, "output_dir", target)
        object.__setattr__(self, "sources", sources)

    @property
    def session_id(self) -> str:
        return self.sources[0].session_id


@dataclass(frozen=True, slots=True)
class ReconstructionExportFile:
    role: str
    relative_path: str
    authority: ReconstructionArtifactAuthority
    size_bytes: int
    sha256: str
    artifact_id: str = ""
    canonical_geometry_authoritative: bool = False
    certified_metrology: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", require_nonempty(self.role, field_name="role"))
        relative = Path(require_nonempty(self.relative_path, field_name="relative_path"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ReconstructionValidationError("Export relative_path must remain inside the export bundle.")
        if not isinstance(self.authority, ReconstructionArtifactAuthority):
            raise TypeError("authority must be ReconstructionArtifactAuthority.")
        size = int(self.size_bytes)
        if size < 0:
            raise ReconstructionValidationError("Export file size_bytes must be non-negative.")
        digest = str(self.sha256).strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ReconstructionValidationError("Export file sha256 must be a 64-character hexadecimal digest.")
        canonical = self.authority is ReconstructionArtifactAuthority.CANONICAL_GEOMETRY
        if bool(self.canonical_geometry_authoritative) != canonical:
            raise ReconstructionValidationError(
                "Export canonical_geometry_authoritative must agree with artifact authority."
            )
        if bool(self.certified_metrology):
            raise ReconstructionValidationError("Phase 8 export files cannot claim certified metrology.")
        object.__setattr__(self, "relative_path", relative.as_posix())
        object.__setattr__(self, "size_bytes", size)
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "artifact_id", str(self.artifact_id).strip())
        object.__setattr__(self, "canonical_geometry_authoritative", canonical)
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "path": self.relative_path,
            "authority": self.authority.value,
            "artifact_id": self.artifact_id,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "canonical_geometry_authoritative": self.canonical_geometry_authoritative,
            "certified_metrology": False,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ReconstructionExportResult:
    export_dir: Path
    manifest_path: Path
    session_id: str
    files: tuple[ReconstructionExportFile, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "export_dir", Path(self.export_dir).resolve())
        object.__setattr__(self, "manifest_path", Path(self.manifest_path).resolve())
        object.__setattr__(self, "session_id", require_nonempty(self.session_id, field_name="session_id"))
        object.__setattr__(self, "files", tuple(self.files))
        digest = str(self.manifest_sha256).strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ReconstructionValidationError("manifest_sha256 must be a 64-character hexadecimal digest.")
        object.__setattr__(self, "manifest_sha256", digest)


def export_reconstruction_bundle(request: ReconstructionExportRequest) -> ReconstructionExportResult:
    """Atomically export the requested reconstruction evidence bundle.

    The function is intentionally backend-free.  It only copies already
    materialized sparse artifacts or invokes the existing dependency-light
    surface/QC writers for already-qualified typed results.
    """

    if not isinstance(request, ReconstructionExportRequest):
        raise TypeError("request must be ReconstructionExportRequest.")
    target = request.output_dir
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent)))
    files: list[ReconstructionExportFile] = []
    try:
        for source in request.sources:
            if isinstance(source, OfflineSparseExportSource):
                files.extend(_export_offline_sparse_source(source, temporary))
            elif isinstance(source, CanonicalSurfaceMeshExportSource):
                files.append(_export_surface_source(source, temporary))
            elif isinstance(source, InspectionReportExportSource):
                files.append(_export_inspection_source(source, temporary))
            else:  # pragma: no cover - request validation makes this unreachable
                raise TypeError(f"Unsupported reconstruction export source: {type(source).__name__}.")

        files.sort(key=lambda item: (item.relative_path, item.role))
        has_canonical = any(item.canonical_geometry_authoritative for item in files)
        manifest = {
            "schema": RECONSTRUCTION_EXPORT_SCHEMA,
            "session_id": request.session_id,
            "file_count": len(files),
            "canonical_geometry_authoritative": has_canonical,
            "certified_metrology": False,
            "authority_model": {
                "sparse_sfm": ReconstructionArtifactAuthority.SUPPORTING_GEOMETRY.value,
                "surface_mesh": ReconstructionArtifactAuthority.CANONICAL_GEOMETRY.value,
                "inspection_qc": ReconstructionArtifactAuthority.INSPECTION_DECISION.value,
                "gaussian_nerf": ReconstructionArtifactAuthority.APPEARANCE_ONLY.value,
            },
            "files": [item.as_dict() for item in files],
        }
        manifest_path = temporary / RECONSTRUCTION_EXPORT_MANIFEST
        _write_json(manifest_path, manifest)
        manifest_sha256 = _sha256_file(manifest_path)
        (temporary / RECONSTRUCTION_EXPORT_CHECKSUM).write_text(
            f"{manifest_sha256}  {RECONSTRUCTION_EXPORT_MANIFEST}\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
        final_manifest = target / RECONSTRUCTION_EXPORT_MANIFEST
        return ReconstructionExportResult(
            export_dir=target,
            manifest_path=final_manifest,
            session_id=request.session_id,
            files=tuple(files),
            manifest_sha256=manifest_sha256,
        )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_reconstruction_export_bundle(path: str | Path) -> tuple[ReconstructionExportFile, ...]:
    """Fail closed if a previously exported bundle or payload file was modified."""

    root = Path(path).expanduser()
    if not root.is_absolute():
        raise ReconstructionValidationError("Reconstruction export path must be absolute.")
    root = root.resolve()
    manifest_path = root / RECONSTRUCTION_EXPORT_MANIFEST if root.is_dir() else root
    if not manifest_path.is_file():
        raise ReconstructionValidationError("Reconstruction export manifest does not exist.")
    checksum_path = manifest_path.parent / RECONSTRUCTION_EXPORT_CHECKSUM
    if not checksum_path.is_file():
        raise ReconstructionValidationError("Reconstruction export manifest checksum sidecar is missing.")
    parts = checksum_path.read_text(encoding="utf-8").strip().split()
    if len(parts) < 2 or parts[1] != RECONSTRUCTION_EXPORT_MANIFEST:
        raise ReconstructionValidationError("Reconstruction export manifest checksum sidecar is malformed.")
    expected_manifest_sha = str(parts[0]).strip().lower()
    if len(expected_manifest_sha) != 64 or any(ch not in "0123456789abcdef" for ch in expected_manifest_sha):
        raise ReconstructionValidationError("Reconstruction export manifest checksum is invalid.")
    if _sha256_file(manifest_path) != expected_manifest_sha:
        raise ReconstructionValidationError("Reconstruction export manifest checksum mismatch.")
    payload = _read_json_mapping(manifest_path, field_name="reconstruction_export_manifest")
    if str(payload.get("schema", "")) != RECONSTRUCTION_EXPORT_SCHEMA:
        raise ReconstructionValidationError("Unsupported reconstruction export schema.")
    if bool(payload.get("certified_metrology", False)):
        raise ReconstructionValidationError("Reconstruction export must not claim certified metrology.")
    root = manifest_path.parent
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ReconstructionValidationError("Reconstruction export manifest has no files.")
    files: list[ReconstructionExportFile] = []
    for raw in raw_files:
        if not isinstance(raw, Mapping):
            raise ReconstructionValidationError("Reconstruction export file records must be objects.")
        item = ReconstructionExportFile(
            role=str(raw.get("role", "")),
            relative_path=str(raw.get("path", "")),
            authority=ReconstructionArtifactAuthority(str(raw.get("authority", ""))),
            artifact_id=str(raw.get("artifact_id", "")),
            size_bytes=int(raw.get("size_bytes", -1)),
            sha256=str(raw.get("sha256", "")),
            canonical_geometry_authoritative=bool(raw.get("canonical_geometry_authoritative", False)),
            certified_metrology=bool(raw.get("certified_metrology", False)),
            metadata=raw.get("metadata", {}) if isinstance(raw.get("metadata", {}), Mapping) else {},
        )
        file_path = (root / item.relative_path).resolve()
        if root not in file_path.parents:
            raise ReconstructionValidationError("Reconstruction export file escapes its bundle directory.")
        if not file_path.is_file():
            raise ReconstructionValidationError(f"Reconstruction export file is missing: {file_path}")
        if file_path.stat().st_size != item.size_bytes:
            raise ReconstructionValidationError(f"Reconstruction export file size changed: {file_path}")
        if _sha256_file(file_path) != item.sha256:
            raise ReconstructionValidationError(f"Reconstruction export file checksum changed: {file_path}")
        files.append(item)
    if bool(payload.get("canonical_geometry_authoritative", False)) != any(
        item.canonical_geometry_authoritative for item in files
    ):
        raise ReconstructionValidationError(
            "Reconstruction export canonical authority summary does not match its file records."
        )
    return tuple(files)


def _export_offline_sparse_source(
    source: OfflineSparseExportSource,
    root: Path,
) -> list[ReconstructionExportFile]:
    geometry_dir = root / "geometry"
    provenance_dir = root / "provenance"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)
    sparse_source = Path(str(source.artifacts["sparse_reconstruction_ply"])).resolve()
    sfm_source = Path(str(source.artifacts["sfm_result_json"])).resolve()
    sparse_target = geometry_dir / "sparse_reconstruction.ply"
    sfm_target = provenance_dir / "sfm_result.json"
    shutil.copy2(sparse_source, sparse_target)
    shutil.copy2(sfm_source, sfm_target)
    sfm_payload = _read_json_mapping(sfm_target, field_name="sfm_result_json")
    gauge = sfm_payload["gauge"]
    assert isinstance(gauge, Mapping)
    common = {
        "scale_status": "arbitrary",
        "unit_to_metre": None,
        "backend_id": str(sfm_payload.get("backend_id", "")),
        "backend_version": str(sfm_payload.get("backend_version", "")),
    }
    return [
        _file_record(
            sparse_target,
            root=root,
            role="sparse_point_cloud",
            authority=ReconstructionArtifactAuthority.SUPPORTING_GEOMETRY,
            metadata={**common, "format": "ply"},
        ),
        _file_record(
            sfm_target,
            root=root,
            role="sparse_sfm_result",
            authority=ReconstructionArtifactAuthority.SUPPORTING_GEOMETRY,
            metadata={**common, "format": "json"},
        ),
    ]


def _export_surface_source(
    source: CanonicalSurfaceMeshExportSource,
    root: Path,
) -> ReconstructionExportFile:
    destination = root / "geometry" / "canonical_surface_mesh.obj"
    write_surface_mesh_obj(source.result, destination)
    return _file_record(
        destination,
        root=root,
        role="canonical_surface_mesh",
        authority=ReconstructionArtifactAuthority.CANONICAL_GEOMETRY,
        artifact_id=source.artifact.artifact_id,
        metadata={
            "format": "obj",
            "provider_id": source.artifact.provider_id,
            "metric_geometry_status": source.result.metric_geometry_status.value,
            "unit_to_metre": source.result.unit_to_metre,
            "relative_scale_uncertainty": source.result.relative_scale_uncertainty,
            "vertex_count": source.result.mesh.vertex_count,
            "face_count": source.result.mesh.face_count,
        },
    )


def _export_inspection_source(
    source: InspectionReportExportSource,
    root: Path,
) -> ReconstructionExportFile:
    destination = root / "qc" / "inspection_report.json"
    write_inspection_report_json(source.report, destination)
    return _file_record(
        destination,
        root=root,
        role="inspection_qc_report",
        authority=ReconstructionArtifactAuthority.INSPECTION_DECISION,
        artifact_id=source.artifact.artifact_id,
        metadata={
            "format": "json",
            "provider_id": source.artifact.provider_id,
            "decision": source.report.decision.value,
            "comparison_metric_available": source.report.comparison_metric_available,
            "reference_unit_to_metre": source.report.reference_unit_to_metre,
            "relative_scale_uncertainty": source.report.relative_scale_uncertainty,
        },
    )


def _file_record(
    path: Path,
    *,
    root: Path,
    role: str,
    authority: ReconstructionArtifactAuthority,
    artifact_id: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> ReconstructionExportFile:
    resolved = path.resolve()
    return ReconstructionExportFile(
        role=role,
        relative_path=resolved.relative_to(root.resolve()).as_posix(),
        authority=authority,
        artifact_id=artifact_id,
        size_bytes=resolved.stat().st_size,
        sha256=_sha256_file(resolved),
        canonical_geometry_authoritative=(authority is ReconstructionArtifactAuthority.CANONICAL_GEOMETRY),
        certified_metrology=False,
        metadata=dict(metadata or {}),
    )


def _required_absolute_file(value: object, *, field_name: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ReconstructionValidationError(f"{field_name} must reference a file.")
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ReconstructionValidationError(f"{field_name} must be an absolute path.")
    path = path.resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ReconstructionValidationError(f"{field_name} does not reference a non-empty file: {path}")
    return path


def _read_json_mapping(path: Path, *, field_name: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"{field_name} is not readable valid JSON.") from exc
    if not isinstance(payload, Mapping):
        raise ReconstructionValidationError(f"{field_name} root must be a JSON object.")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "RECONSTRUCTION_EXPORT_CHECKSUM",
    "RECONSTRUCTION_EXPORT_MANIFEST",
    "RECONSTRUCTION_EXPORT_SCHEMA",
    "CanonicalSurfaceMeshExportSource",
    "InspectionReportExportSource",
    "OfflineSparseExportSource",
    "ReconstructionExportFile",
    "ReconstructionExportRequest",
    "ReconstructionExportResult",
    "ReconstructionExportSource",
    "export_reconstruction_bundle",
    "verify_reconstruction_export_bundle",
]
