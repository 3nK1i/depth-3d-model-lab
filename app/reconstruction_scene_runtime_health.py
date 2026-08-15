"""Phase 9.3I scene-binding performance/resource lifecycle presentation.

This module is deliberately presentation-only.  It records UI-side scene binding
latency and resource counts that already exist in the Reconstruction workspace;
it never probes a backend, changes geometry authority, or establishes metric scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ReconstructionSceneBindPlan:
    previous_scene_id: str
    next_scene_id: str
    same_scene_reload: bool
    fit_after_bind: bool
    preserve_selected_object_id: str

    @property
    def preserve_view(self) -> bool:
        return bool(self.same_scene_reload and self.next_scene_id)


def build_reconstruction_scene_bind_plan(
    previous_scene_id: object,
    next_scene_id: object,
    *,
    selected_object_id: object = "",
) -> ReconstructionSceneBindPlan:
    """Return the bounded UX policy for one persistent-scene bind.

    Reloading the exact same scene preserves the user's camera and selected object;
    binding a different scene performs one final Fit after all resources are ready.
    """

    previous = str(previous_scene_id or "").strip()
    next_id = str(next_scene_id or "").strip()
    selected = str(selected_object_id or "").strip()
    same = bool(previous and next_id and previous == next_id)
    return ReconstructionSceneBindPlan(
        previous_scene_id=previous,
        next_scene_id=next_id,
        same_scene_reload=same,
        fit_after_bind=bool(next_id and not same),
        preserve_selected_object_id=selected if same else "",
    )


@dataclass(frozen=True)
class ReconstructionSceneRuntimeHealth:
    scene_id: str
    bind_latency_ms: float
    bind_count: int
    same_scene_reload_count: int
    canonical_vertex_count: int
    canonical_face_count: int
    canonical_camera_count: int
    semantic_object_count: int
    mesh_visual_reused: bool
    view_preserved: bool
    selection_preserved: bool
    canonical_geometry_authoritative: bool
    certified_metrology: bool

    def __post_init__(self) -> None:
        if not isfinite(float(self.bind_latency_ms)) or float(self.bind_latency_ms) < 0.0:
            raise ValueError("bind_latency_ms must be finite and non-negative.")
        for name in (
            "bind_count",
            "same_scene_reload_count",
            "canonical_vertex_count",
            "canonical_face_count",
            "canonical_camera_count",
            "semantic_object_count",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.same_scene_reload_count > self.bind_count:
            raise ValueError("same_scene_reload_count cannot exceed bind_count.")
        if self.certified_metrology:
            raise ValueError("Phase 9.3I runtime health cannot claim certified metrology.")

    @property
    def bind_text(self) -> str:
        if not self.scene_id:
            return "No persistent scene"
        return f"{self.bind_latency_ms:.1f} ms UI bind • {self.bind_count:,} total"

    @property
    def resources_text(self) -> str:
        if not self.scene_id:
            return "No canonical viewer resources"
        return (
            f"{self.canonical_vertex_count:,} vertices • {self.canonical_face_count:,} faces • "
            f"{self.canonical_camera_count:,} cameras • {self.semantic_object_count:,} objects"
        )

    @property
    def lifecycle_text(self) -> str:
        if not self.scene_id:
            return "Idle • no scene-bound resources"
        if self.view_preserved:
            details = ["Same-scene reload", "view preserved"]
            if self.mesh_visual_reused:
                details.append("mesh reused")
            if self.selection_preserved:
                details.append("selection preserved")
            return " • ".join(details)
        return "New scene bind • one final Fit" if self.canonical_vertex_count or self.canonical_camera_count else "Scene bound • no fit target"

    @property
    def authority_text(self) -> str:
        if not self.scene_id:
            return "Diagnostics only • no scene authority"
        if self.canonical_geometry_authoritative:
            return "Diagnostics only • canonical authority unchanged • metrology not certified"
        return "Diagnostics only • no canonical promotion • metrology not certified"

    def as_dict(self) -> dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "bind_latency_ms": float(self.bind_latency_ms),
            "bind_count": int(self.bind_count),
            "same_scene_reload_count": int(self.same_scene_reload_count),
            "canonical_vertex_count": int(self.canonical_vertex_count),
            "canonical_face_count": int(self.canonical_face_count),
            "canonical_camera_count": int(self.canonical_camera_count),
            "semantic_object_count": int(self.semantic_object_count),
            "mesh_visual_reused": bool(self.mesh_visual_reused),
            "view_preserved": bool(self.view_preserved),
            "selection_preserved": bool(self.selection_preserved),
            "canonical_geometry_authoritative": bool(self.canonical_geometry_authoritative),
            "certified_metrology": False,
            "diagnostics_only": True,
        }


def build_reconstruction_scene_runtime_health(
    *,
    scene_id: object,
    bind_latency_ms: float,
    bind_count: int,
    same_scene_reload_count: int,
    canonical_vertex_count: int,
    canonical_face_count: int,
    canonical_camera_count: int,
    semantic_object_count: int,
    mesh_visual_reused: bool,
    view_preserved: bool,
    selection_preserved: bool,
    canonical_geometry_authoritative: bool,
) -> ReconstructionSceneRuntimeHealth:
    return ReconstructionSceneRuntimeHealth(
        scene_id=str(scene_id or "").strip(),
        bind_latency_ms=float(bind_latency_ms),
        bind_count=int(bind_count),
        same_scene_reload_count=int(same_scene_reload_count),
        canonical_vertex_count=int(canonical_vertex_count),
        canonical_face_count=int(canonical_face_count),
        canonical_camera_count=int(canonical_camera_count),
        semantic_object_count=int(semantic_object_count),
        mesh_visual_reused=bool(mesh_visual_reused),
        view_preserved=bool(view_preserved),
        selection_preserved=bool(selection_preserved),
        canonical_geometry_authoritative=bool(canonical_geometry_authoritative),
        certified_metrology=False,
    )


__all__ = [
    "ReconstructionSceneBindPlan",
    "ReconstructionSceneRuntimeHealth",
    "build_reconstruction_scene_bind_plan",
    "build_reconstruction_scene_runtime_health",
]
