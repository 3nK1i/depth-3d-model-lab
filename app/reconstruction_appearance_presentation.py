"""Phase 8.8F1 appearance-representation presentation contracts.

F1 is a UI/application presentation layer over already-persisted Phase 8.7
appearance artifacts.  It does not train, render, probe, launch, or import any
Gaussian/NeRF runtime.  Canonical geometry remains the reconstruction geometry
authority; neural artifacts remain appearance-only context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_APPEARANCE_KIND_LABELS = {
    "gaussian_scene": "Gaussian",
    "nerf_scene": "NeRF",
}


def _value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).strip()


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        try:
            return dict(value)
        except (TypeError, ValueError):
            return {}
    return {}


def _humanize(value: str) -> str:
    return " ".join(part.capitalize() for part in str(value).replace("-", "_").split("_") if part)


@dataclass(frozen=True, slots=True)
class AppearanceRepresentationEntry:
    artifact_id: str
    representation_kind: str
    label: str
    provider_id: str
    backend_id: str
    backend_version: str
    render_capable: bool
    metric_geometry_status: str
    state_text: str
    authority_text: str = "Appearance only"
    canonical_geometry_authoritative: bool = False
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        if not str(self.artifact_id).strip():
            raise ValueError("Appearance representation requires artifact_id.")
        kind = str(self.representation_kind).strip().lower()
        if not kind:
            raise ValueError("Appearance representation requires a representation kind.")
        object.__setattr__(self, "representation_kind", kind)
        object.__setattr__(self, "label", str(self.label).strip() or _humanize(kind))
        object.__setattr__(self, "provider_id", str(self.provider_id).strip())
        object.__setattr__(self, "backend_id", str(self.backend_id).strip())
        object.__setattr__(self, "backend_version", str(self.backend_version).strip())
        object.__setattr__(self, "metric_geometry_status", str(self.metric_geometry_status).strip().lower() or "unknown")
        object.__setattr__(self, "state_text", str(self.state_text).strip())
        if bool(self.canonical_geometry_authoritative):
            raise ValueError("F1 appearance representations cannot claim canonical geometry authority.")
        if bool(self.certified_metrology):
            raise ValueError("F1 appearance representations cannot claim certified metrology.")
        if str(self.authority_text).strip().lower() != "appearance only":
            raise ValueError("F1 appearance representations must remain appearance only.")
        object.__setattr__(self, "authority_text", "Appearance only")
        object.__setattr__(self, "canonical_geometry_authoritative", False)
        object.__setattr__(self, "certified_metrology", False)

    @property
    def provider_text(self) -> str:
        backend = self.backend_id or self.provider_id or "Unknown provider"
        if self.backend_version:
            return f"{backend} • v{self.backend_version}"
        return backend

    @property
    def render_text(self) -> str:
        return "Declared render-capable" if self.render_capable else "Render unavailable"

    @property
    def scale_text(self) -> str:
        status = self.metric_geometry_status
        if status == "arbitrary":
            return "Arbitrary scene gauge • appearance only"
        if status in {"provisional_metric", "validated_metric"}:
            return f"{_humanize(status)} context • appearance only"
        return "Scene scale context unavailable • appearance only"


@dataclass(frozen=True, slots=True)
class ReconstructionAppearancePresentation:
    scene_id: str
    session_id: str
    entries: tuple[AppearanceRepresentationEntry, ...]
    missing_kinds: tuple[str, ...]
    scene_state: str
    canonical_geometry_authoritative: bool
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        if not str(self.session_id).strip():
            raise ValueError("F1 appearance presentation requires session identity.")
        entries = tuple(self.entries)
        if any(not isinstance(item, AppearanceRepresentationEntry) for item in entries):
            raise TypeError("entries must contain AppearanceRepresentationEntry values.")
        if len({item.artifact_id for item in entries}) != len(entries):
            raise ValueError("Appearance artifact IDs must be unique.")
        if bool(self.certified_metrology):
            raise ValueError("F1 cannot claim certified metrology.")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "missing_kinds", tuple(str(value).strip().lower() for value in self.missing_kinds if str(value).strip()))
        object.__setattr__(self, "scene_state", str(self.scene_state).strip().lower() or "unknown")
        object.__setattr__(self, "canonical_geometry_authoritative", bool(self.canonical_geometry_authoritative))
        object.__setattr__(self, "certified_metrology", False)

    @property
    def available_text(self) -> str:
        if not self.entries:
            return "No neural appearance artifacts"
        return " • ".join(item.label for item in self.entries)

    @property
    def state_text(self) -> str:
        if not self.entries:
            return "No persisted appearance representation"
        if self.missing_kinds:
            missing = ", ".join(_APPEARANCE_KIND_LABELS.get(value, _humanize(value)) for value in self.missing_kinds)
            return f"Appearance metadata available • {missing} missing"
        return "Appearance metadata ready • visualization available in F2/F3"

    @property
    def authority_text(self) -> str:
        return "Appearance only • canonical geometry remains authoritative"

    def entry_by_artifact_id(self, artifact_id: str) -> AppearanceRepresentationEntry | None:
        needle = str(artifact_id).strip()
        return next((entry for entry in self.entries if entry.artifact_id == needle), None)


def _artifact_dict(artifact: Any) -> dict[str, Any]:
    method = getattr(artifact, "as_dict", None)
    if callable(method):
        payload = method()
        if isinstance(payload, Mapping):
            return dict(payload)
    return {
        "artifact_id": getattr(artifact, "artifact_id", ""),
        "kind": _value(getattr(artifact, "kind", "")),
        "authority": _value(getattr(artifact, "authority", "")),
        "provider_id": getattr(artifact, "provider_id", ""),
        "metadata": getattr(artifact, "metadata", {}),
        "payload_snapshot": getattr(artifact, "payload_snapshot", {}),
        "canonical_geometry_authoritative": getattr(artifact, "canonical_geometry_authoritative", False),
        "certified_metrology": getattr(artifact, "certified_metrology", False),
    }


def _entry_from_artifact(artifact: Any) -> AppearanceRepresentationEntry | None:
    raw = _artifact_dict(artifact)
    kind = _value(raw.get("kind", "")).lower()
    if kind not in _APPEARANCE_KIND_LABELS:
        return None
    authority = _value(raw.get("authority", "")).lower()
    if authority != "appearance_only":
        raise ValueError("Persisted neural representation changed appearance-only authority.")
    if bool(raw.get("canonical_geometry_authoritative", False)) or bool(raw.get("certified_metrology", False)):
        raise ValueError("Persisted neural representation claims forbidden geometry/metrology authority.")

    metadata = _mapping(raw.get("metadata", {}))
    snapshot = _mapping(raw.get("payload_snapshot", {}))
    # Payload snapshots carry backend detail; normalized artifact metadata is
    # the fallback. F1 only reads already-persisted values.
    details = dict(metadata)
    details.update({key: value for key, value in snapshot.items() if value not in (None, "")})
    render_capable = bool(details.get("render_capable", False))
    if kind == "gaussian_scene":
        stage_state = str(details.get("optimization_state", "")).strip()
    else:
        stage_state = str(details.get("training_state", "")).strip()
    state_text = _humanize(stage_state) if stage_state else ("Artifact ready" if render_capable else "Artifact metadata only")
    if render_capable:
        state_text = f"{state_text} • render capability declared"

    return AppearanceRepresentationEntry(
        artifact_id=str(raw.get("artifact_id", "")),
        representation_kind=kind,
        label=_APPEARANCE_KIND_LABELS[kind],
        provider_id=str(raw.get("provider_id", "")),
        backend_id=str(details.get("backend_id", "")),
        backend_version=str(details.get("backend_version", "")),
        render_capable=render_capable,
        metric_geometry_status=str(details.get("metric_geometry_status", "unknown")),
        state_text=state_text,
        authority_text="Appearance only",
        canonical_geometry_authoritative=False,
        certified_metrology=False,
    )


def build_reconstruction_appearance_presentation(scene: Any) -> ReconstructionAppearancePresentation | None:
    """Build F1 metadata controls from an existing persistent reconstruction scene.

    This function deliberately performs no filesystem probing and no backend
    availability/resource checks.  Rich failure/resource handling belongs to F5.
    """

    if scene is None:
        return None
    session_id = str(getattr(scene, "session_id", "")).strip()
    if not session_id:
        raise ValueError("Persistent scene does not expose session identity.")
    scene_id = str(getattr(scene, "scene_id", "")).strip()
    scene_state = _value(getattr(scene, "state", "unknown")).lower()
    canonical_authoritative = bool(getattr(scene, "canonical_geometry_authoritative", False))
    if bool(getattr(scene, "certified_metrology", False)):
        raise ValueError("Persistent scene unexpectedly claims certified metrology.")

    entries: list[AppearanceRepresentationEntry] = []
    present_kinds: set[str] = set()
    for artifact in tuple(getattr(scene, "artifacts", ()) or ()):
        entry = _entry_from_artifact(artifact)
        if entry is None:
            continue
        entries.append(entry)
        present_kinds.add(entry.representation_kind)

    order = {"gaussian_scene": 0, "nerf_scene": 1}
    entries.sort(key=lambda item: (order.get(item.representation_kind, 100), item.label, item.artifact_id))
    expected = ("gaussian_scene", "nerf_scene")
    missing = tuple(kind for kind in expected if kind not in present_kinds)
    return ReconstructionAppearancePresentation(
        scene_id=scene_id,
        session_id=session_id,
        entries=tuple(entries),
        missing_kinds=missing,
        scene_state=scene_state,
        canonical_geometry_authoritative=canonical_authoritative,
        certified_metrology=False,
    )


def empty_appearance_representation_text() -> dict[str, str]:
    return {
        "available": "No persistent appearance scene",
        "provider": "—",
        "artifact": "—",
        "render": "—",
        "scale": "—",
        "state": "Waiting for persistent reconstruction scene",
        "authority": "Appearance only • canonical geometry remains authoritative",
    }
