"""Phase 8.8F4 application-level appearance representation switching.

F4 coordinates the already-qualified F2 Gaussian and F3 NeRF static preview
presentations. It does not render neural representations or launch isolated
backends. F5 adds bounded resource/fallback UX around this explicit switching
path. Canonical geometry remains authoritative regardless of the active view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_CANONICAL_KIND = "canonical_geometry"
_GAUSSIAN_KIND = "gaussian_scene"
_NERF_KIND = "nerf_scene"
_APPEARANCE_AUTHORITY = "Appearance only • canonical geometry remains authoritative"


@dataclass(frozen=True, slots=True)
class RepresentationSwitchTarget:
    """A lightweight F4 target derived from persisted presentation state."""

    artifact_id: str
    representation_kind: str
    label: str
    ready: bool
    state_text: str
    authority_text: str = _APPEARANCE_AUTHORITY
    canonical_geometry_authoritative: bool = False
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        kind = str(self.representation_kind).strip().lower()
        if kind not in {_GAUSSIAN_KIND, _NERF_KIND}:
            raise ValueError(f"Unsupported appearance representation kind: {kind or '<empty>'}")
        if not str(self.artifact_id).strip():
            raise ValueError("F4 representation switching requires artifact identity.")
        if bool(self.canonical_geometry_authoritative):
            raise ValueError("F4 appearance switching cannot claim canonical geometry authority.")
        if bool(self.certified_metrology):
            raise ValueError("F4 appearance switching cannot claim certified metrology.")
        object.__setattr__(self, "artifact_id", str(self.artifact_id).strip())
        object.__setattr__(self, "representation_kind", kind)
        object.__setattr__(self, "label", str(self.label).strip() or _label_for_kind(kind))
        object.__setattr__(self, "ready", bool(self.ready))
        object.__setattr__(self, "state_text", str(self.state_text).strip())
        object.__setattr__(self, "authority_text", _APPEARANCE_AUTHORITY)
        object.__setattr__(self, "canonical_geometry_authoritative", False)
        object.__setattr__(self, "certified_metrology", False)

    @property
    def active_text(self) -> str:
        return f"{self.label} appearance • appearance only"

    @property
    def unavailable_text(self) -> str:
        return f"{self.label} unavailable • canonical geometry retained"


def _label_for_kind(kind: str) -> str:
    if kind == _GAUSSIAN_KIND:
        return "Gaussian"
    if kind == _NERF_KIND:
        return "NeRF"
    return "Appearance"


def _presentation_for_kind(
    kind: str,
    gaussian_presentation: Any,
    nerf_presentation: Any,
) -> Any:
    if kind == _GAUSSIAN_KIND:
        return gaussian_presentation
    if kind == _NERF_KIND:
        return nerf_presentation
    return None


def build_representation_switch_target(
    selected: Any,
    *,
    gaussian_presentation: Any = None,
    nerf_presentation: Any = None,
) -> RepresentationSwitchTarget | None:
    """Resolve the selected F1 artifact to an already-qualified F2/F3 target.

    ``ready`` reflects only persisted preview presentation readiness. It does
    not choose another target when the selected representation is unavailable;
    F5 reports the explicit canonical fallback without neural cross-substitution.
    """

    if selected is None:
        return None
    kind = str(getattr(selected, "representation_kind", "")).strip().lower()
    if kind not in {_GAUSSIAN_KIND, _NERF_KIND}:
        return None
    artifact_id = str(getattr(selected, "artifact_id", "")).strip()
    if not artifact_id:
        return None
    presentation = _presentation_for_kind(kind, gaussian_presentation, nerf_presentation)
    presentation_id = str(getattr(presentation, "artifact_id", "")).strip() if presentation is not None else ""
    matches = bool(presentation is not None and presentation_id == artifact_id)
    ready = bool(matches and getattr(presentation, "ready", False))
    state_text = str(getattr(presentation, "state_text", "")).strip() if matches else ""
    if not state_text:
        state_text = f"{_label_for_kind(kind)} preview unavailable"
    return RepresentationSwitchTarget(
        artifact_id=artifact_id,
        representation_kind=kind,
        label=str(getattr(selected, "label", "")).strip() or _label_for_kind(kind),
        ready=ready,
        state_text=state_text,
    )


def canonical_view_text() -> str:
    return "Canonical geometry"


__all__ = [
    "RepresentationSwitchTarget",
    "build_representation_switch_target",
    "canonical_view_text",
]
