"""Phase 8.8F2 Gaussian-splatting visualization presentation contracts.

F2 visualizes an already-produced Gaussian preview render from a persisted
Phase 8.7 appearance artifact.  It deliberately does not import or execute a
Gaussian runtime, parse the heavy Gaussian scene payload, retrain, or promote
appearance to canonical geometry. Phase 8.8F4 coordinates this qualified
preview with F3; F5 reports bounded resource/fallback UX without backend execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


_GAUSSIAN_KIND = "gaussian_scene"
_APPEARANCE_AUTHORITY = "appearance_only"
_PREVIEW_FILENAME = "preview.png"
_QUALIFICATION_METRICS_FILENAME = "qualification_metrics.json"


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


def _scene_gaussian_artifact(scene: Any) -> Any | None:
    direct = getattr(scene, "gaussian_artifact", None)
    if direct is not None:
        return direct
    for artifact in tuple(getattr(scene, "artifacts", ()) or ()):
        raw = _artifact_dict(artifact)
        if _value(raw.get("kind", "")).lower() == _GAUSSIAN_KIND:
            return artifact
    return None


def _path_inside(root: Path, child: Path) -> bool:
    try:
        child.relative_to(root)
        return True
    except ValueError:
        return False


def _read_preview_from_metrics(artifact_dir: Path) -> tuple[Path | None, str, str]:
    metrics = artifact_dir / _QUALIFICATION_METRICS_FILENAME
    if not metrics.is_file():
        return None, "", ""
    try:
        payload = json.loads(metrics.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "", "qualification metrics unreadable"
    if not isinstance(payload, Mapping):
        return None, "", "qualification metrics invalid"
    if not bool(payload.get("render_written", False)):
        return None, "", "qualification metrics do not confirm a preview render"
    candidate_text = str(payload.get("preview_path", "")).strip()
    if not candidate_text:
        return None, "", "qualification metrics omit preview path"
    candidate = Path(candidate_text).expanduser()
    if not candidate.is_absolute():
        candidate = artifact_dir / candidate
    try:
        candidate = candidate.resolve()
    except OSError:
        return None, "", "qualification preview path cannot be resolved"
    if not _path_inside(artifact_dir, candidate):
        return None, "", "qualification preview path escapes Gaussian artifact directory"
    return candidate, "qualification metrics", ""


def resolve_gaussian_preview_path(artifact_dir: str | Path) -> tuple[Path | None, str, str]:
    """Resolve a backend-authored Gaussian preview without launching a backend.

    The qualified gsplat and Graphdeco workers both write ``preview.png`` inside
    their artifact directory.  When qualification metrics are present, F2 uses
    their explicit ``preview_path`` but refuses paths outside the artifact root.
    The conventional artifact-local preview is a safe compatibility fallback.
    """

    root = Path(artifact_dir).expanduser()
    if not root.is_absolute():
        return None, "", "Gaussian artifact directory is not absolute"
    try:
        root = root.resolve()
    except OSError:
        return None, "", "Gaussian artifact directory cannot be resolved"
    if not root.is_dir():
        return None, "", "Gaussian artifact directory is unavailable"

    metrics_candidate, metrics_source, metrics_error = _read_preview_from_metrics(root)
    if metrics_error:
        return None, "", metrics_error
    if metrics_candidate is not None:
        if metrics_candidate.is_file() and metrics_candidate.stat().st_size > 0:
            return metrics_candidate, metrics_source, ""
        return None, metrics_source, "Gaussian preview referenced by qualification metrics is missing"

    candidate = (root / _PREVIEW_FILENAME).resolve()
    if candidate.is_file() and candidate.stat().st_size > 0:
        return candidate, "artifact-local preview", ""
    return None, "", "Gaussian backend preview render is unavailable"


@dataclass(frozen=True, slots=True)
class GaussianVisualizationPresentation:
    scene_id: str
    session_id: str
    artifact_id: str
    provider_id: str
    backend_id: str
    backend_version: str
    preview_path: str
    preview_source: str
    gaussian_count: int
    render_capable: bool
    metric_geometry_status: str
    ready: bool
    state_text: str
    authority_text: str = "Appearance only • canonical geometry remains authoritative"
    canonical_geometry_authoritative: bool = False
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        if not str(self.session_id).strip():
            raise ValueError("Gaussian visualization requires session identity.")
        if not str(self.artifact_id).strip():
            raise ValueError("Gaussian visualization requires artifact identity.")
        if bool(self.canonical_geometry_authoritative):
            raise ValueError("F2 Gaussian visualization cannot claim canonical geometry authority.")
        if bool(self.certified_metrology):
            raise ValueError("F2 Gaussian visualization cannot claim certified metrology.")
        object.__setattr__(self, "provider_id", str(self.provider_id).strip())
        object.__setattr__(self, "backend_id", str(self.backend_id).strip())
        object.__setattr__(self, "backend_version", str(self.backend_version).strip())
        object.__setattr__(self, "preview_path", str(self.preview_path).strip())
        object.__setattr__(self, "preview_source", str(self.preview_source).strip())
        object.__setattr__(self, "gaussian_count", max(0, int(self.gaussian_count)))
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
        if self.gaussian_count > 0:
            return f"{self.gaussian_count:,} Gaussians • appearance only"
        return "Gaussian appearance scene"

    @property
    def scale_text(self) -> str:
        if self.metric_geometry_status == "arbitrary":
            return "Arbitrary scene gauge • appearance only"
        if self.metric_geometry_status in {"provisional_metric", "validated_metric"}:
            return f"{self.metric_geometry_status.replace('_', ' ').title()} context • appearance only"
        return "Scale context unavailable • appearance only"


def build_gaussian_visualization_presentation(scene: Any) -> GaussianVisualizationPresentation | None:
    """Build F2 state from a persisted Gaussian artifact and its preview render."""

    if scene is None:
        return None
    session_id = str(getattr(scene, "session_id", "")).strip()
    if not session_id:
        raise ValueError("Persistent scene does not expose session identity.")
    artifact = _scene_gaussian_artifact(scene)
    if artifact is None:
        return None

    raw = _artifact_dict(artifact)
    if _value(raw.get("kind", "")).lower() != _GAUSSIAN_KIND:
        raise ValueError("F2 received a non-Gaussian appearance artifact.")
    if _value(raw.get("authority", "")).lower() != _APPEARANCE_AUTHORITY:
        raise ValueError("Persisted Gaussian artifact changed appearance-only authority.")
    if bool(raw.get("canonical_geometry_authoritative", False)):
        raise ValueError("Persisted Gaussian artifact claims forbidden canonical authority.")
    if bool(raw.get("certified_metrology", False)):
        raise ValueError("Persisted Gaussian artifact claims forbidden certified metrology.")

    metadata = _mapping(raw.get("metadata", {}))
    snapshot = _mapping(raw.get("payload_snapshot", {}))
    details = dict(metadata)
    details.update({key: value for key, value in snapshot.items() if value not in (None, "")})

    artifact_dir = str(details.get("artifact_dir", "")).strip()
    render_capable = bool(details.get("render_capable", False))
    preview_path: Path | None = None
    preview_source = ""
    reason = ""
    if not render_capable:
        reason = "Gaussian artifact is not declared render-capable"
    elif not artifact_dir:
        reason = "Gaussian artifact directory is unavailable in persisted metadata"
    else:
        preview_path, preview_source, reason = resolve_gaussian_preview_path(artifact_dir)

    ready = bool(render_capable and preview_path is not None)
    if ready:
        state = "Backend-authored Gaussian preview ready • static visualization in F2"
    else:
        state = reason or "Gaussian preview unavailable"

    return GaussianVisualizationPresentation(
        scene_id=str(getattr(scene, "scene_id", "")).strip(),
        session_id=session_id,
        artifact_id=str(raw.get("artifact_id", "")).strip(),
        provider_id=str(raw.get("provider_id", "")).strip(),
        backend_id=str(details.get("backend_id", "")).strip(),
        backend_version=str(details.get("backend_version", "")).strip(),
        preview_path=str(preview_path) if preview_path is not None else "",
        preview_source=preview_source,
        gaussian_count=int(details.get("gaussian_count", 0) or 0),
        render_capable=render_capable,
        metric_geometry_status=str(details.get("metric_geometry_status", "unknown")),
        ready=ready,
        state_text=state,
        canonical_geometry_authoritative=False,
        certified_metrology=False,
    )


def empty_gaussian_visualization_text() -> dict[str, str]:
    return {
        "provider": "—",
        "scene": "—",
        "preview": "—",
        "scale": "—",
        "state": "No persistent Gaussian representation",
        "authority": "Appearance only • canonical geometry remains authoritative",
    }
