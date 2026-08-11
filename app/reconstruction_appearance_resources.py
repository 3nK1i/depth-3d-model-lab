"""Phase 8.8F5 resource and fallback presentation for appearance views.

F5 makes resource state and canonical fallback behavior explicit in the
Reconstruction UI.  It consumes the application's existing startup hardware
snapshot plus persisted F2/F3/F4 presentation state; it does not import, probe,
or launch an external Gaussian/NeRF runtime and it never promotes appearance to
canonical geometry, measurement, QC, or reference authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.hardware.gpu_info import HardwareInfo, detect_hardware


_CANONICAL_KIND = "canonical_geometry"
_GAUSSIAN_KIND = "gaussian_scene"
_NERF_KIND = "nerf_scene"
_APPEARANCE_STAGES = {
    "gaussian_appearance": "Gaussian appearance",
    "nerf_appearance": "NeRF appearance",
}
_APPEARANCE_AUTHORITY = "Appearance only • canonical geometry remains authoritative"


def _value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).strip()


def _humanize(value: str) -> str:
    return " ".join(
        part.capitalize()
        for part in str(value).replace("-", "_").split("_")
        if part
    )


def _hardware_text(hardware: HardwareInfo) -> tuple[str, str, str]:
    device = hardware.short_label
    total = hardware.total_memory_gib
    free = hardware.free_memory_gib
    if hardware.has_cuda_device:
        if hardware.cuda_ready:
            if total is not None and free is not None:
                resources = f"CUDA ready • {free:.1f}/{total:.1f} GB startup free/total"
            elif total is not None:
                resources = f"CUDA ready • {total:.1f} GB total VRAM"
            else:
                resources = "CUDA ready • VRAM snapshot unavailable"
        else:
            resources = "CUDA unavailable in app runtime • static previews remain supported"
    else:
        resources = "CPU runtime • static previews remain supported"
    details = hardware.details
    return device, resources, details


def _appearance_fallback_evidence(scene: Any) -> tuple[str, str]:
    if scene is None:
        return "No persistent scene", ""
    items: list[str] = []
    for evidence in tuple(getattr(scene, "fallback_evidence", ()) or ()):
        stage = _value(getattr(evidence, "stage", "")).lower()
        if stage not in _APPEARANCE_STAGES:
            continue
        state = _value(getattr(evidence, "state", "unknown")).lower() or "unknown"
        reason = str(getattr(evidence, "reason", "")).strip()
        fallback_used = bool(getattr(evidence, "fallback_used", False))
        label = _APPEARANCE_STAGES[stage]
        state_label = _humanize(state)
        if reason:
            items.append(f"{label}: {state_label} • {reason}")
        elif fallback_used:
            items.append(f"{label}: fallback used")
        else:
            items.append(f"{label}: {state_label}")
    if not items:
        return "No appearance-stage fallback", ""
    if len(items) == 1:
        text = items[0]
        if len(text) > 76:
            text = text[:73].rstrip() + "…"
        return text, items[0]
    return f"{len(items)} appearance fallbacks recorded", "\n".join(items)


@dataclass(frozen=True, slots=True)
class AppearanceResourceFallbackPresentation:
    """Application-facing F5 resource/fallback state.

    Existing static preview presentation is intentionally independent from CUDA
    readiness: a previously authored PNG remains displayable on CPU.  Resource
    status is therefore advisory for the app runtime and never causes an
    automatic Gaussian↔NeRF substitution.
    """

    device_text: str
    resources_text: str
    resource_details: str
    presentation_text: str
    fallback_text: str
    pipeline_text: str
    pipeline_details: str
    state_text: str
    state_details: str
    selected_kind: str
    active_view: str
    preview_ready: bool
    fallback_active: bool
    resource_snapshot_available: bool
    cross_representation_fallback: bool = False
    appearance_canonical_geometry_authoritative: bool = False
    certified_metrology: bool = False
    authority_text: str = _APPEARANCE_AUTHORITY

    def __post_init__(self) -> None:
        active = str(self.active_view).strip().lower() or _CANONICAL_KIND
        selected = str(self.selected_kind).strip().lower()
        if active not in {_CANONICAL_KIND, _GAUSSIAN_KIND, _NERF_KIND}:
            raise ValueError("F5 active_view must be canonical or a qualified appearance kind.")
        if selected and selected not in {_GAUSSIAN_KIND, _NERF_KIND}:
            raise ValueError("F5 selected_kind must be a qualified appearance kind when present.")
        if bool(self.cross_representation_fallback):
            raise ValueError("F5 must not silently cross-fallback between neural representations.")
        if bool(self.appearance_canonical_geometry_authoritative):
            raise ValueError("F5 appearance resources cannot claim canonical geometry authority.")
        if bool(self.certified_metrology):
            raise ValueError("F5 appearance resources cannot claim certified metrology.")
        object.__setattr__(self, "device_text", str(self.device_text).strip() or "Unknown device")
        object.__setattr__(self, "resources_text", str(self.resources_text).strip())
        object.__setattr__(self, "resource_details", str(self.resource_details).strip())
        object.__setattr__(self, "presentation_text", str(self.presentation_text).strip())
        object.__setattr__(self, "fallback_text", "Canonical geometry • no neural cross-fallback")
        object.__setattr__(self, "pipeline_text", str(self.pipeline_text).strip())
        object.__setattr__(self, "pipeline_details", str(self.pipeline_details).strip())
        object.__setattr__(self, "state_text", str(self.state_text).strip())
        object.__setattr__(self, "state_details", str(self.state_details).strip())
        object.__setattr__(self, "selected_kind", selected)
        object.__setattr__(self, "active_view", active)
        object.__setattr__(self, "preview_ready", bool(self.preview_ready))
        object.__setattr__(self, "fallback_active", bool(self.fallback_active))
        object.__setattr__(self, "resource_snapshot_available", bool(self.resource_snapshot_available))
        object.__setattr__(self, "cross_representation_fallback", False)
        object.__setattr__(self, "appearance_canonical_geometry_authoritative", False)
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "authority_text", _APPEARANCE_AUTHORITY)


def build_appearance_resource_fallback_presentation(
    scene: Any,
    *,
    selected: Any = None,
    switch_target: Any = None,
    active_view: str = _CANONICAL_KIND,
    failure_reason: str = "",
    hardware: HardwareInfo | None = None,
) -> AppearanceResourceFallbackPresentation:
    """Build F5 UX from existing state without launching neural backends.

    ``hardware`` defaults to the application's cached startup hardware snapshot.
    The snapshot is advisory only.  Existing F2/F3 static previews are not
    disabled when CUDA is unavailable or memory is constrained.
    """

    hardware = hardware or detect_hardware()
    if not isinstance(hardware, HardwareInfo):
        raise TypeError("hardware must be HardwareInfo.")
    device_text, resources_text, resource_details = _hardware_text(hardware)
    pipeline_text, pipeline_details = _appearance_fallback_evidence(scene)

    selected_kind = str(getattr(selected, "representation_kind", "")).strip().lower()
    selected_label = str(getattr(selected, "label", "")).strip() or _humanize(selected_kind)
    target_ready = bool(switch_target is not None and getattr(switch_target, "ready", False))
    active = str(active_view).strip().lower() or _CANONICAL_KIND
    failure = str(failure_reason).strip()

    if scene is None:
        presentation_text = "Canonical viewport • no appearance scene"
        state_text = "Waiting for persistent reconstruction scene"
        state_details = "F5 has no persisted appearance state to evaluate."
        fallback_active = False
    elif active in {_GAUSSIAN_KIND, _NERF_KIND}:
        label = "Gaussian" if active == _GAUSSIAN_KIND else "NeRF"
        presentation_text = f"{label} static preview • no neural runtime required"
        state_text = "Appearance view active • canonical authority retained"
        state_details = (
            "F5 is presenting an already-authored static preview. The external neural backend remains isolated and is not launched."
        )
        fallback_active = False
    elif failure:
        presentation_text = "Canonical viewport • appearance request failed safely"
        state_text = f"Fallback active • {failure}"
        state_details = (
            "Canonical geometry was retained after the selected appearance view could not be presented. "
            "F5 does not substitute another neural representation."
        )
        fallback_active = True
    elif selected_kind and target_ready:
        presentation_text = f"{selected_label} static preview ready • backend-free display"
        state_text = "Ready • activate selected representation"
        state_details = (
            "The selected persisted preview can be displayed without starting the external authoring runtime."
        )
        fallback_active = False
    elif selected_kind:
        target_state = str(getattr(switch_target, "state_text", "")).strip()
        reason = target_state or f"{selected_label} preview unavailable"
        presentation_text = "Canonical viewport • selected preview unavailable"
        state_text = f"Fallback active • {reason}"
        state_details = (
            "Canonical geometry is retained. No Gaussian↔NeRF cross-fallback is performed; the selected representation remains explicit."
        )
        fallback_active = True
    elif pipeline_details or pipeline_text != "No appearance-stage fallback":
        presentation_text = "Canonical viewport • appearance pipeline degraded"
        state_text = "Appearance unavailable • canonical geometry retained"
        state_details = pipeline_details or pipeline_text
        fallback_active = True
    else:
        presentation_text = "Canonical viewport • no appearance representation"
        state_text = "No persisted appearance representation available"
        state_details = "Canonical geometry remains available and authoritative."
        fallback_active = False

    return AppearanceResourceFallbackPresentation(
        device_text=device_text,
        resources_text=resources_text,
        resource_details=resource_details,
        presentation_text=presentation_text,
        fallback_text="Canonical geometry • no neural cross-fallback",
        pipeline_text=pipeline_text,
        pipeline_details=pipeline_details,
        state_text=state_text,
        state_details=state_details,
        selected_kind=selected_kind,
        active_view=active,
        preview_ready=target_ready,
        fallback_active=fallback_active,
        resource_snapshot_available=True,
    )


def empty_appearance_resource_fallback_text() -> dict[str, str]:
    return {
        "device": "—",
        "resources": "Startup resource snapshot unavailable",
        "presentation": "Canonical viewport",
        "fallback": "Canonical geometry • no neural cross-fallback",
        "pipeline": "No persistent scene",
        "state": "Waiting for persistent reconstruction scene",
    }


__all__ = [
    "AppearanceResourceFallbackPresentation",
    "build_appearance_resource_fallback_presentation",
    "empty_appearance_resource_fallback_text",
]
