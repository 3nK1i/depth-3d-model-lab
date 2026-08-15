"""Presentation-only scene-layer availability for Phase 9.3F.

Layer visibility never changes reconstruction, semantic, measurement or QC
authority.  This module only tells the Reconstruction sidebar which optional
canonical-scene controls currently have materialized evidence behind them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReconstructionSceneLayerAvailability:
    canonical_surface: bool
    semantic_objects: bool
    semantic_labels: bool
    semantic_selection: bool
    inspection_evidence: bool

    def __post_init__(self) -> None:
        for name in (
            "canonical_surface",
            "semantic_objects",
            "semantic_labels",
            "semantic_selection",
            "inspection_evidence",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))
        if self.semantic_labels and not self.semantic_objects:
            raise ValueError("Semantic labels require semantic object context.")
        if self.semantic_selection and not self.semantic_objects:
            raise ValueError("Semantic selection requires semantic object context.")
        if self.inspection_evidence and not self.semantic_selection:
            raise ValueError("Inspection evidence requires selected semantic-object context.")

    @property
    def certified_metrology(self) -> bool:
        return False

    @property
    def canonical_geometry_authoritative(self) -> bool:
        return False

    def as_dict(self) -> dict[str, bool]:
        return {
            "canonical_surface": self.canonical_surface,
            "semantic_objects": self.semantic_objects,
            "semantic_labels": self.semantic_labels,
            "semantic_selection": self.semantic_selection,
            "inspection_evidence": self.inspection_evidence,
            "visibility_only": True,
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        }


def build_reconstruction_scene_layer_availability(
    *,
    canonical_surface_loaded: bool,
    semantic_objects_available: bool,
    canonical_semantics_loaded: bool,
    semantic_selection_active: bool,
    inspection_evidence_available: bool,
) -> ReconstructionSceneLayerAvailability:
    """Build fail-closed availability without inventing scene evidence."""

    semantics = bool(semantic_objects_available and canonical_semantics_loaded)
    selection = bool(semantics and semantic_selection_active)
    inspection = bool(selection and inspection_evidence_available)
    return ReconstructionSceneLayerAvailability(
        canonical_surface=bool(canonical_surface_loaded),
        semantic_objects=bool(semantic_objects_available),
        semantic_labels=semantics,
        semantic_selection=selection,
        inspection_evidence=inspection,
    )


__all__ = [
    "ReconstructionSceneLayerAvailability",
    "build_reconstruction_scene_layer_availability",
]
