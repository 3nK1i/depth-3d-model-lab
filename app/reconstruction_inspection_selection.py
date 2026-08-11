"""Generic scene/object selection presentation for Phase 8.8E1.

E1 does not perform inspection, measurement, registration, reference matching,
or QC.  It binds the already-active reconstruction scene and the existing D5
semantic-object selection into one immutable, object-agnostic downstream target
contract that later E-series UI steps can consume without duplicating selectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.offline_reconstruction_semantics import OfflineSemanticPresentation


class ReconstructionInspectionSelectionScope(str, Enum):
    NONE = "none"
    SCENE = "scene"
    OBJECT = "object"


@dataclass(frozen=True, slots=True)
class ReconstructionInspectionSelection:
    """UI/application selection contract for one active reconstruction target."""

    scope: ReconstructionInspectionSelectionScope
    scene_id: str = ""
    session_id: str = ""
    scene_source: str = ""
    scene_availability: str = "none"
    canonical_geometry_authoritative: bool = False
    object_id: str = ""
    class_name: str = ""
    semantic_role: str = ""
    semantic_source: str = ""
    metric_geometry_status: str = "unknown"
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ReconstructionInspectionSelectionScope):
            raise TypeError("scope must be ReconstructionInspectionSelectionScope.")
        for field_name in (
            "scene_id",
            "session_id",
            "scene_source",
            "scene_availability",
            "object_id",
            "class_name",
            "semantic_role",
            "semantic_source",
            "metric_geometry_status",
        ):
            object.__setattr__(self, field_name, str(getattr(self, field_name)).strip())
        status = self.metric_geometry_status.lower() or "unknown"
        if status not in {"arbitrary", "provisional_metric", "validated_metric", "unknown"}:
            raise ValueError(f"Unsupported metric geometry status: {status!r}")
        object.__setattr__(self, "metric_geometry_status", status)
        object.__setattr__(
            self,
            "canonical_geometry_authoritative",
            bool(self.canonical_geometry_authoritative),
        )
        if bool(self.certified_metrology):
            raise ValueError("Phase 8.8E1 selection cannot claim certified metrology.")
        object.__setattr__(self, "certified_metrology", False)

        has_scene = bool(self.scene_id or self.session_id)
        has_object_identity = bool(self.object_id or self.class_name or self.semantic_role)
        if self.scope is ReconstructionInspectionSelectionScope.NONE:
            if has_scene or has_object_identity:
                raise ValueError("NONE selection cannot carry scene/object identity.")
        elif self.scope is ReconstructionInspectionSelectionScope.SCENE:
            if not has_scene:
                raise ValueError("SCENE selection requires scene or session identity.")
            if has_object_identity:
                raise ValueError("SCENE selection cannot carry object identity.")
        elif self.scope is ReconstructionInspectionSelectionScope.OBJECT:
            if not has_scene:
                raise ValueError("OBJECT selection requires scene or session identity.")
            if not self.object_id or not self.class_name or not self.semantic_role:
                raise ValueError("OBJECT selection requires object ID, class, and semantic role.")
            if not self.semantic_source:
                raise ValueError("OBJECT selection requires semantic selection provenance.")

    @property
    def has_scene(self) -> bool:
        return self.scope is not ReconstructionInspectionSelectionScope.NONE

    @property
    def has_object(self) -> bool:
        return self.scope is ReconstructionInspectionSelectionScope.OBJECT

    @property
    def scope_text(self) -> str:
        if self.scope is ReconstructionInspectionSelectionScope.OBJECT:
            return "Object"
        if self.scope is ReconstructionInspectionSelectionScope.SCENE:
            return "Scene"
        return "—"

    @property
    def scene_text(self) -> str:
        if self.scene_id:
            return _compact_identity(self.scene_id)
        if self.session_id:
            return f"Session {_compact_identity(self.session_id)}"
        return "No reconstruction scene"

    @property
    def object_text(self) -> str:
        if not self.has_object:
            return "—"
        role = self.semantic_role.replace("_", " ").title()
        return f"{self.class_name} • {role}"

    @property
    def geometry_text(self) -> str:
        if self.canonical_geometry_authoritative:
            return "Canonical scene geometry available"
        if self.has_object:
            return "Semantic support geometry available"
        if self.has_scene:
            return "Scene context available"
        return "No geometry target"

    @property
    def state_text(self) -> str:
        if self.scope is ReconstructionInspectionSelectionScope.OBJECT:
            return "Object target selected"
        if self.scope is ReconstructionInspectionSelectionScope.SCENE:
            return "Scene target selected"
        return "No inspection target"

    @property
    def authority_text(self) -> str:
        return "Selection only • measurement/QC not evaluated"

    @property
    def scene_tooltip(self) -> str:
        if not self.has_scene:
            return "No reconstruction scene/session is currently available for downstream inspection."
        parts = []
        if self.scene_id:
            parts.append(f"scene_id={self.scene_id}")
        if self.session_id:
            parts.append(f"session_id={self.session_id}")
        if self.scene_source:
            parts.append(f"source={self.scene_source}")
        if self.scene_availability:
            parts.append(f"availability={self.scene_availability}")
        return " • ".join(parts)

    @property
    def object_tooltip(self) -> str:
        if not self.has_object:
            return "No semantic object is selected; the active scene remains the E1 target."
        return (
            f"object_id={self.object_id} • class={self.class_name} • "
            f"role={self.semantic_role} • source={self.semantic_source}"
        )


def build_reconstruction_inspection_selection(
    *,
    scene_id: str = "",
    session_id: str = "",
    scene_source: str = "",
    scene_availability: str = "none",
    canonical_geometry_authoritative: bool = False,
    semantic_presentation: OfflineSemanticPresentation | None = None,
    selected_object_index: int | None = None,
) -> ReconstructionInspectionSelection:
    """Build one downstream target from current scene + existing D5 selection.

    The function deliberately accepts generic scene identity and the normalized
    D5 semantic presentation.  It does not know connector/pin classes and does
    not inspect geometry, calculate measurements, run QC, or load references.
    """

    scene_id = str(scene_id).strip()
    session_id = str(session_id).strip()
    scene_source = str(scene_source).strip()
    scene_availability = str(scene_availability).strip().lower() or "none"

    if semantic_presentation is not None and not isinstance(
        semantic_presentation, OfflineSemanticPresentation
    ):
        raise TypeError("semantic_presentation must be OfflineSemanticPresentation or None.")

    semantic_session = (
        semantic_presentation.session_id if semantic_presentation is not None else ""
    )
    if session_id and semantic_session and session_id != semantic_session:
        raise ValueError("Semantic selection session does not match the active reconstruction scene.")
    if not session_id and semantic_session:
        session_id = semantic_session
    if not scene_source and semantic_presentation is not None:
        scene_source = semantic_presentation.source

    if selected_object_index is not None:
        if semantic_presentation is None:
            raise ValueError("Object selection requires semantic presentation context.")
        index = int(selected_object_index)
        if index < 0 or index >= len(semantic_presentation.objects):
            raise ValueError("Selected semantic object index is outside the presentation bounds.")
        item = semantic_presentation.objects[index]
        return ReconstructionInspectionSelection(
            scope=ReconstructionInspectionSelectionScope.OBJECT,
            scene_id=scene_id,
            session_id=session_id,
            scene_source=scene_source,
            scene_availability=scene_availability,
            canonical_geometry_authoritative=bool(canonical_geometry_authoritative),
            object_id=item.object_id,
            class_name=item.class_name,
            semantic_role=item.semantic_role,
            semantic_source=semantic_presentation.source,
            metric_geometry_status=item.metric_geometry_status,
            certified_metrology=False,
        )

    has_scene_context = bool(scene_id or session_id)
    if has_scene_context:
        status = (
            semantic_presentation.metric_geometry_status
            if semantic_presentation is not None
            else "unknown"
        )
        return ReconstructionInspectionSelection(
            scope=ReconstructionInspectionSelectionScope.SCENE,
            scene_id=scene_id,
            session_id=session_id,
            scene_source=scene_source,
            scene_availability=scene_availability,
            canonical_geometry_authoritative=bool(canonical_geometry_authoritative),
            metric_geometry_status=status,
            certified_metrology=False,
        )

    return ReconstructionInspectionSelection(
        scope=ReconstructionInspectionSelectionScope.NONE,
        scene_availability=scene_availability,
        certified_metrology=False,
    )


def _compact_identity(value: str, *, limit: int = 28) -> str:
    text = str(value).strip()
    if len(text) <= limit:
        return text
    keep = max(6, (limit - 1) // 2)
    return f"{text[:keep]}…{text[-keep:]}"
