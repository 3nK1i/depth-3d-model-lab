"""Phase 8.8F3 NeRF visualization presentation contracts.

F3 visualizes an already-produced NeRF preview render from a persisted Phase
8.7 appearance artifact. It deliberately does not import or execute a NeRF
runtime, parse checkpoints/configuration for rendering, retrain, or promote
appearance to canonical geometry. Whole-representation switching remains Phase
8.8F4 and resource/fallback policy remains Phase 8.8F5.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


_NERF_KIND = "nerf_scene"
_APPEARANCE_AUTHORITY = "appearance_only"
_PREVIEW_FILENAME = "preview.png"
_RUNTIME_EVIDENCE_FILENAME = "e2_runtime_evidence.json"
_HELDOUT_RENDER_MANIFEST_FILENAME = "heldout_render_manifest.json"


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
        "canonical_geometry_authoritative": getattr(
            artifact, "canonical_geometry_authoritative", False
        ),
        "certified_metrology": getattr(artifact, "certified_metrology", False),
    }


def _scene_nerf_artifact(scene: Any) -> Any | None:
    direct = getattr(scene, "nerf_artifact", None)
    if direct is not None:
        return direct
    for artifact in tuple(getattr(scene, "artifacts", ()) or ()):
        raw = _artifact_dict(artifact)
        if _value(raw.get("kind", "")).lower() == _NERF_KIND:
            return artifact
    return None


def _path_inside(root: Path, child: Path) -> bool:
    try:
        child.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_candidate(root: Path, value: object, *, source: str) -> tuple[Path | None, str]:
    candidate_text = str(value or "").strip()
    if not candidate_text:
        return None, f"{source} omits preview path"
    candidate = Path(candidate_text).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        candidate = candidate.resolve()
    except OSError:
        return None, f"{source} preview path cannot be resolved"
    if not _path_inside(root, candidate):
        return None, f"{source} preview path escapes NeRF artifact directory"
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        return None, f"NeRF preview referenced by {source} is missing"
    return candidate, ""


def _preview_from_runtime_evidence(root: Path) -> tuple[Path | None, str, str]:
    evidence = root / _RUNTIME_EVIDENCE_FILENAME
    if not evidence.is_file():
        return None, "", ""
    try:
        payload = json.loads(evidence.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "", "runtime evidence unreadable"
    if not isinstance(payload, Mapping):
        return None, "", "runtime evidence invalid"
    candidate, error = _safe_candidate(root, payload.get("preview_path"), source="runtime evidence")
    if error:
        return None, "", error
    return candidate, "runtime evidence", ""


def _preview_from_heldout_manifest(root: Path) -> tuple[Path | None, str, str]:
    manifest = root / _HELDOUT_RENDER_MANIFEST_FILENAME
    if not manifest.is_file():
        return None, "", ""
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "", "held-out render manifest unreadable"
    if not isinstance(payload, Mapping):
        return None, "", "held-out render manifest invalid"
    renders = payload.get("renders")
    if not isinstance(renders, list) or not renders:
        return None, "", "held-out render manifest has no rendered views"
    first = renders[0]
    if not isinstance(first, Mapping):
        return None, "", "held-out render manifest contains an invalid render entry"
    candidate, error = _safe_candidate(
        root,
        first.get("rendered_image_path"),
        source="held-out render manifest",
    )
    if error:
        return None, "", error
    return candidate, "held-out render manifest", ""


def resolve_nerf_preview_path(artifact_dir: str | Path) -> tuple[Path | None, str, str]:
    """Resolve an existing backend-authored NeRF image without launching a backend."""

    root = Path(artifact_dir).expanduser()
    if not root.is_absolute():
        return None, "", "NeRF artifact directory is not absolute"
    try:
        root = root.resolve()
    except OSError:
        return None, "", "NeRF artifact directory cannot be resolved"
    if not root.is_dir():
        return None, "", "NeRF artifact directory is unavailable"

    evidence_candidate, evidence_source, evidence_error = _preview_from_runtime_evidence(root)
    if evidence_error:
        return None, "", evidence_error
    if evidence_candidate is not None:
        return evidence_candidate, evidence_source, ""

    manifest_candidate, manifest_source, manifest_error = _preview_from_heldout_manifest(root)
    if manifest_error:
        return None, "", manifest_error
    if manifest_candidate is not None:
        return manifest_candidate, manifest_source, ""

    candidate = (root / _PREVIEW_FILENAME).resolve()
    if candidate.is_file() and candidate.stat().st_size > 0:
        return candidate, "artifact-local preview", ""
    return None, "", "NeRF backend preview render is unavailable"


@dataclass(frozen=True, slots=True)
class NerfVisualizationPresentation:
    scene_id: str
    session_id: str
    artifact_id: str
    provider_id: str
    backend_id: str
    backend_version: str
    preview_path: str
    preview_source: str
    input_frame_count: int
    training_steps: int
    render_capable: bool
    metric_geometry_status: str
    ready: bool
    state_text: str
    authority_text: str = "Appearance only • canonical geometry remains authoritative"
    canonical_geometry_authoritative: bool = False
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        if not str(self.session_id).strip():
            raise ValueError("NeRF visualization requires session identity.")
        if not str(self.artifact_id).strip():
            raise ValueError("NeRF visualization requires artifact identity.")
        if bool(self.canonical_geometry_authoritative):
            raise ValueError("F3 NeRF visualization cannot claim canonical geometry authority.")
        if bool(self.certified_metrology):
            raise ValueError("F3 NeRF visualization cannot claim certified metrology.")
        object.__setattr__(self, "provider_id", str(self.provider_id).strip())
        object.__setattr__(self, "backend_id", str(self.backend_id).strip())
        object.__setattr__(self, "backend_version", str(self.backend_version).strip())
        object.__setattr__(self, "preview_path", str(self.preview_path).strip())
        object.__setattr__(self, "preview_source", str(self.preview_source).strip())
        object.__setattr__(self, "input_frame_count", max(0, int(self.input_frame_count)))
        object.__setattr__(self, "training_steps", max(0, int(self.training_steps)))
        object.__setattr__(self, "render_capable", bool(self.render_capable))
        object.__setattr__(
            self,
            "metric_geometry_status",
            str(self.metric_geometry_status).strip().lower() or "unknown",
        )
        object.__setattr__(self, "ready", bool(self.ready))
        object.__setattr__(self, "state_text", str(self.state_text).strip())
        object.__setattr__(
            self,
            "authority_text",
            "Appearance only • canonical geometry remains authoritative",
        )
        object.__setattr__(self, "canonical_geometry_authoritative", False)
        object.__setattr__(self, "certified_metrology", False)

    @property
    def provider_text(self) -> str:
        backend = self.backend_id or self.provider_id or "Unknown provider"
        return f"{backend} • v{self.backend_version}" if self.backend_version else backend

    @property
    def preview_text(self) -> str:
        if not self.preview_path:
            return "No backend preview render"
        name = Path(self.preview_path).name
        source = self.preview_source or "artifact"
        return f"{name} • {source}"

    @property
    def scene_text(self) -> str:
        parts: list[str] = []
        if self.input_frame_count > 0:
            parts.append(f"{self.input_frame_count:,} frames")
        if self.training_steps > 0:
            parts.append(f"{self.training_steps:,} steps")
        if parts:
            return " • ".join(parts) + " • appearance only"
        return "NeRF appearance scene"

    @property
    def scale_text(self) -> str:
        if self.metric_geometry_status == "arbitrary":
            return "Arbitrary scene gauge • appearance only"
        if self.metric_geometry_status in {"provisional_metric", "validated_metric"}:
            return f"{self.metric_geometry_status.replace('_', ' ').title()} context • appearance only"
        return "Scale context unavailable • appearance only"


def build_nerf_visualization_presentation(scene: Any) -> NerfVisualizationPresentation | None:
    """Build F3 state from a persisted NeRF artifact and an existing render."""

    if scene is None:
        return None
    session_id = str(getattr(scene, "session_id", "")).strip()
    if not session_id:
        raise ValueError("Persistent scene does not expose session identity.")
    artifact = _scene_nerf_artifact(scene)
    if artifact is None:
        return None

    raw = _artifact_dict(artifact)
    if _value(raw.get("kind", "")).lower() != _NERF_KIND:
        raise ValueError("F3 received a non-NeRF appearance artifact.")
    if _value(raw.get("authority", "")).lower() != _APPEARANCE_AUTHORITY:
        raise ValueError("Persisted NeRF artifact changed appearance-only authority.")
    if bool(raw.get("canonical_geometry_authoritative", False)):
        raise ValueError("Persisted NeRF artifact claims forbidden canonical authority.")
    if bool(raw.get("certified_metrology", False)):
        raise ValueError("Persisted NeRF artifact claims forbidden certified metrology.")

    metadata = _mapping(raw.get("metadata", {}))
    snapshot = _mapping(raw.get("payload_snapshot", {}))
    details = dict(metadata)
    details.update({key: value for key, value in snapshot.items() if value not in (None, "")})

    render_capable = bool(details.get("render_capable", False))
    artifact_dir = str(details.get("artifact_dir", "")).strip()
    preview_path: Path | None = None
    preview_source = ""
    reason = ""
    if not render_capable:
        reason = "NeRF artifact is not declared render-capable"
    elif not artifact_dir:
        reason = "NeRF artifact directory is unavailable"
    else:
        preview_path, preview_source, reason = resolve_nerf_preview_path(artifact_dir)

    ready = bool(render_capable and preview_path is not None)
    state = (
        "Backend-authored NeRF preview ready • static visualization in F3"
        if ready
        else (reason or "NeRF preview unavailable")
    )

    return NerfVisualizationPresentation(
        scene_id=str(getattr(scene, "scene_id", "")).strip(),
        session_id=session_id,
        artifact_id=str(raw.get("artifact_id", "")),
        provider_id=str(raw.get("provider_id", "")),
        backend_id=str(details.get("backend_id", "")),
        backend_version=str(details.get("backend_version", "")),
        preview_path=str(preview_path) if preview_path is not None else "",
        preview_source=preview_source,
        input_frame_count=int(details.get("input_frame_count", 0) or 0),
        training_steps=int(details.get("training_steps", 0) or 0),
        render_capable=render_capable,
        metric_geometry_status=str(details.get("metric_geometry_status", "unknown")),
        ready=ready,
        state_text=state,
        authority_text="Appearance only • canonical geometry remains authoritative",
        canonical_geometry_authoritative=False,
        certified_metrology=False,
    )


def empty_nerf_visualization_text() -> dict[str, str]:
    return {
        "provider": "—",
        "scene": "—",
        "preview": "—",
        "scale": "—",
        "state": "No persistent NeRF representation",
        "authority": "Appearance only • canonical geometry remains authoritative",
    }
