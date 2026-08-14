"""Phase 9.2B reusable segmentation-project and class-catalog foundation.

A segmentation project owns stable semantic class IDs/names across frames and
sessions.  Per-frame :mod:`core.perception.semantic_workspace` instances remain
immutable review snapshots and can import the project catalog without changing
mask, geometry, metric, or measurement authority.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
from uuid import uuid4

from core.contracts.validation import ContractValidationError, require_nonempty
from core.perception.semantic_workspace import (
    SemanticClassCatalog,
    SemanticClassDefinition,
    SemanticClassOrigin,
    SemanticWorkspace,
)


SEGMENTATION_PROJECT_SCHEMA_VERSION = 1
_PROJECT_FILE_SUFFIX = ".segmentation-project.json"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validated_utc_timestamp(value: str, *, field_name: str) -> str:
    text = require_nonempty(value, field_name=field_name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractValidationError(f"{field_name} must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None:
        raise ContractValidationError(f"{field_name} must include a timezone offset.")
    return parsed.astimezone(timezone.utc).isoformat()


def _slugify_class_name(class_name: str) -> str:
    text = require_nonempty(class_name, field_name="class_name").strip().casefold()
    slug = _SLUG_RE.sub("_", text).strip("_")
    if not slug:
        raise ContractValidationError("class_name does not contain a usable instance-name token.")
    return slug


@dataclass(frozen=True, slots=True)
class SemanticInstanceNamingPolicy:
    """Deterministic human-friendly instance naming for project classes."""

    separator: str = "_"
    zero_pad: int = 3

    def __post_init__(self) -> None:
        separator = str(self.separator)
        if not separator or any(char.isspace() for char in separator):
            raise ContractValidationError("instance-name separator must be non-empty and contain no whitespace.")
        if isinstance(self.zero_pad, bool) or int(self.zero_pad) < 1 or int(self.zero_pad) > 8:
            raise ContractValidationError("zero_pad must be an integer in [1, 8].")
        object.__setattr__(self, "separator", separator)
        object.__setattr__(self, "zero_pad", int(self.zero_pad))

    def suggest(
        self,
        class_name: str,
        *,
        existing_names: Iterable[str] = (),
    ) -> str:
        base = _slugify_class_name(class_name)
        used = {str(name).strip().casefold() for name in existing_names if str(name).strip()}
        index = 1
        while True:
            candidate = f"{base}{self.separator}{index:0{self.zero_pad}d}"
            if candidate.casefold() not in used:
                return candidate
            index += 1

    def as_dict(self) -> dict[str, Any]:
        return {"separator": self.separator, "zero_pad": self.zero_pad}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticInstanceNamingPolicy":
        return cls(
            separator=str(payload.get("separator", "_")),
            zero_pad=int(payload.get("zero_pad", 3)),
        )


@dataclass(frozen=True, slots=True)
class SegmentationProject:
    """Reusable semantic class catalog shared across Auto-Segment workspaces."""

    project_id: str
    project_name: str
    class_catalog: SemanticClassCatalog
    description: str = ""
    provenance: str = "user_project"
    naming_policy: SemanticInstanceNamingPolicy = SemanticInstanceNamingPolicy()
    created_at_utc: str = ""
    updated_at_utc: str = ""
    schema_version: int = SEGMENTATION_PROJECT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "project_id", require_nonempty(self.project_id, field_name="project_id")
        )
        object.__setattr__(
            self, "project_name", require_nonempty(self.project_name, field_name="project_name")
        )
        if not isinstance(self.class_catalog, SemanticClassCatalog):
            raise TypeError("class_catalog must be SemanticClassCatalog.")
        if not isinstance(self.naming_policy, SemanticInstanceNamingPolicy):
            raise TypeError("naming_policy must be SemanticInstanceNamingPolicy.")
        object.__setattr__(self, "description", str(self.description).strip())
        object.__setattr__(
            self, "provenance", require_nonempty(self.provenance, field_name="provenance")
        )
        created = self.created_at_utc or _utc_now_iso()
        updated = self.updated_at_utc or created
        created = _validated_utc_timestamp(created, field_name="created_at_utc")
        updated = _validated_utc_timestamp(updated, field_name="updated_at_utc")
        if datetime.fromisoformat(updated) < datetime.fromisoformat(created):
            raise ContractValidationError("updated_at_utc cannot precede created_at_utc.")
        object.__setattr__(self, "created_at_utc", created)
        object.__setattr__(self, "updated_at_utc", updated)
        if int(self.schema_version) != SEGMENTATION_PROJECT_SCHEMA_VERSION:
            raise ContractValidationError(
                f"Unsupported segmentation-project schema version: {self.schema_version}."
            )
        object.__setattr__(self, "schema_version", int(self.schema_version))

    @property
    def is_geometry_authority(self) -> bool:
        return False

    @property
    def is_metric_authority(self) -> bool:
        return False

    @property
    def is_measurement_authority(self) -> bool:
        return False

    def class_by_id(self, class_id: int) -> SemanticClassDefinition | None:
        return self.class_catalog.by_id(class_id)

    def class_by_name(self, class_name: str) -> SemanticClassDefinition | None:
        return self.class_catalog.by_name(class_name)

    def suggest_instance_name(
        self,
        class_id: int,
        *,
        existing_names: Iterable[str] = (),
    ) -> str:
        definition = self.class_by_id(class_id)
        if definition is None:
            raise ContractValidationError(f"Unknown project class_id: {class_id}.")
        return self.naming_policy.suggest(
            definition.class_name,
            existing_names=existing_names,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "project_name": self.project_name,
            "description": self.description,
            "provenance": self.provenance,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "naming_policy": self.naming_policy.as_dict(),
            "classes": [
                {
                    "class_id": definition.class_id,
                    "class_name": definition.class_name,
                    "origin": definition.origin.value,
                    "source_model_id": definition.source_model_id,
                    "description": definition.description,
                }
                for definition in self.class_catalog.classes
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SegmentationProject":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("Segmentation project root must be a mapping.")
        classes_payload = payload.get("classes", ())
        if not isinstance(classes_payload, list):
            raise ContractValidationError("Segmentation project classes must be a JSON array.")
        classes: list[SemanticClassDefinition] = []
        for item in classes_payload:
            if not isinstance(item, Mapping):
                raise ContractValidationError("Every segmentation project class must be an object.")
            try:
                origin = SemanticClassOrigin(str(item["origin"]))
                classes.append(
                    SemanticClassDefinition(
                        class_id=int(item["class_id"]),
                        class_name=str(item["class_name"]),
                        origin=origin,
                        source_model_id=str(item.get("source_model_id", "")),
                        description=str(item.get("description", "")),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ContractValidationError("Invalid segmentation project class record.") from exc
        naming_payload = payload.get("naming_policy", {})
        if not isinstance(naming_payload, Mapping):
            raise ContractValidationError("naming_policy must be a JSON object.")
        try:
            return cls(
                project_id=str(payload["project_id"]),
                project_name=str(payload["project_name"]),
                description=str(payload.get("description", "")),
                provenance=str(payload.get("provenance", "user_project")),
                class_catalog=SemanticClassCatalog(tuple(classes)),
                naming_policy=SemanticInstanceNamingPolicy.from_dict(naming_payload),
                created_at_utc=str(payload["created_at_utc"]),
                updated_at_utc=str(payload["updated_at_utc"]),
                schema_version=int(payload.get("schema_version", SEGMENTATION_PROJECT_SCHEMA_VERSION)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractValidationError("Invalid segmentation project record.") from exc


def create_segmentation_project(
    project_name: str,
    *,
    project_id: str | None = None,
    description: str = "",
    provenance: str = "user_project",
    class_catalog: SemanticClassCatalog | None = None,
    naming_policy: SemanticInstanceNamingPolicy | None = None,
    created_at_utc: str | None = None,
) -> SegmentationProject:
    timestamp = created_at_utc or _utc_now_iso()
    return SegmentationProject(
        project_id=project_id or f"segmentation-project-{uuid4()}",
        project_name=project_name,
        description=description,
        provenance=provenance,
        class_catalog=class_catalog or SemanticClassCatalog(),
        naming_policy=naming_policy or SemanticInstanceNamingPolicy(),
        created_at_utc=timestamp,
        updated_at_utc=timestamp,
    )


def create_segmentation_project_from_model_names(
    project_name: str,
    names: Mapping[int, str] | Iterable[tuple[int, str]],
    *,
    source_model_id: str,
    project_id: str | None = None,
    description: str = "",
    provenance: str = "model_seeded_project",
    created_at_utc: str | None = None,
) -> SegmentationProject:
    catalog = SemanticClassCatalog.from_model_names(
        names,
        source_model_id=require_nonempty(source_model_id, field_name="source_model_id"),
    )
    return create_segmentation_project(
        project_name,
        project_id=project_id,
        description=description,
        provenance=provenance,
        class_catalog=catalog,
        created_at_utc=created_at_utc,
    )


def add_project_class(
    project: SegmentationProject,
    class_name: str,
    *,
    description: str = "",
    updated_at_utc: str | None = None,
) -> tuple[SegmentationProject, SemanticClassDefinition]:
    catalog, definition = project.class_catalog.with_custom_class(
        class_name,
        description=description,
    )
    if catalog is project.class_catalog:
        return project, definition
    return (
        replace(
            project,
            class_catalog=catalog,
            updated_at_utc=updated_at_utc or _utc_now_iso(),
        ),
        definition,
    )


def rename_project_class(
    project: SegmentationProject,
    class_id: int,
    new_name: str,
    *,
    description: str | None = None,
    updated_at_utc: str | None = None,
) -> SegmentationProject:
    """Rename a custom/imported project class while preserving its stable class ID.

    Model-defined classes are immutable.  Reclassifying a model observation should
    create/choose a project class instead of silently changing the model ontology.
    """

    current = project.class_catalog.by_id(class_id)
    if current is None:
        raise ContractValidationError(f"Unknown project class_id: {class_id}.")
    if current.origin is SemanticClassOrigin.MODEL:
        raise ContractValidationError(
            "Model-defined classes are immutable; create or assign a custom project class instead."
        )
    clean_name = require_nonempty(new_name, field_name="new_name")
    collision = project.class_catalog.by_name(clean_name)
    if collision is not None and collision.class_id != current.class_id:
        raise ContractValidationError(
            f"class_name {clean_name!r} is already assigned to class_id {collision.class_id}."
        )
    replacement = SemanticClassDefinition(
        class_id=current.class_id,
        class_name=clean_name,
        origin=current.origin,
        source_model_id=current.source_model_id,
        description=current.description if description is None else str(description).strip(),
    )
    classes = tuple(
        replacement if item.class_id == current.class_id else item
        for item in project.class_catalog.classes
    )
    return replace(
        project,
        class_catalog=SemanticClassCatalog(classes),
        updated_at_utc=updated_at_utc or _utc_now_iso(),
    )


def merge_project_catalog_into_workspace(
    workspace: SemanticWorkspace,
    project: SegmentationProject,
) -> SemanticWorkspace:
    """Add project classes to a workspace without touching objects or authority.

    ID/name collisions must represent the same definition; ambiguous remapping is
    rejected instead of silently changing YOLO/project class semantics.
    """

    catalog = workspace.class_catalog
    for definition in project.class_catalog.classes:
        catalog = catalog.with_class(definition)
    if catalog == workspace.class_catalog:
        return workspace
    return SemanticWorkspace(
        workspace_id=workspace.workspace_id,
        frame_reference=workspace.frame_reference,
        class_catalog=catalog,
        objects=workspace.objects,
        revision=workspace.revision + 1,
    )


def segmentation_project_to_json(project: SegmentationProject) -> str:
    if not isinstance(project, SegmentationProject):
        raise TypeError("project must be SegmentationProject.")
    return json.dumps(project.as_dict(), indent=2, sort_keys=True) + "\n"


def save_segmentation_project(project: SegmentationProject, path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(segmentation_project_to_json(project), encoding="utf-8")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def load_segmentation_project(path: str | Path) -> SegmentationProject:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ContractValidationError(f"Segmentation project file does not exist: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractValidationError("Could not parse segmentation project JSON.") from exc
    return SegmentationProject.from_dict(payload)


class SegmentationProjectStore:
    """Filesystem-backed reusable project store with path-safe deterministic names."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def _path_for_id(self, project_id: str) -> Path:
        key = require_nonempty(project_id, field_name="project_id")
        digest = sha256(key.encode("utf-8")).hexdigest()
        return self.root / f"{digest}{_PROJECT_FILE_SUFFIX}"

    def save(self, project: SegmentationProject) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return save_segmentation_project(project, self._path_for_id(project.project_id))

    def contains(self, project_id: str) -> bool:
        return self._path_for_id(project_id).is_file()

    def load(self, project_id: str) -> SegmentationProject:
        return load_segmentation_project(self._path_for_id(project_id))

    def list_projects(self) -> tuple[SegmentationProject, ...]:
        if not self.root.is_dir():
            return ()
        projects = [load_segmentation_project(path) for path in self.root.glob(f"*{_PROJECT_FILE_SUFFIX}")]
        return tuple(sorted(projects, key=lambda item: (item.project_name.casefold(), item.project_id)))
