from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from app.offline_reconstruction_semantics import (
    MAX_SEMANTIC_OBJECT_PRESENTATIONS,
    semantic_bounds_segments_in_viewer,
    semantic_centroid_in_viewer,
    semantic_presentation_from_payload,
    semantic_presentation_from_persistent_scene,
)


def _object(index: int, *, role: str = "connector_pin", class_name: str = "pin") -> dict[str, object]:
    x = float(index) * 0.1
    return {
        "object_id": f"semantic-{index:04d}",
        "class_name": class_name,
        "semantic_role": role,
        "confidence": 0.93,
        "vertex_count": 120 + index,
        "face_count": 80 + index,
        "centroid_reconstruction": [x, 0.0, 2.0],
        "bounds_min_reconstruction": [x - 0.04, -0.08, 1.95],
        "bounds_max_reconstruction": [x + 0.04, 0.08, 2.05],
        "orientation_reliable": True,
        "observation_ids": [f"obs-{index}-0", f"obs-{index}-1"],
        "mean_vertex_quality": 0.88,
        "median_vertex_uncertainty_reconstruction_units": 0.05,
        "maximum_vertex_uncertainty_reconstruction_units": 0.08,
        "metric_geometry_status": "arbitrary",
        "certified_metrology": False,
    }


def _payload(count: int = 3) -> dict[str, object]:
    objects = [_object(index) for index in range(count)]
    if objects:
        objects[0] = _object(0, role="connector_body", class_name="connector_body")
    return {
        "session_id": "phase8-8d5-semantic-test",
        "metric_geometry_status": "arbitrary",
        "certified_metrology": False,
        "statistics": {
            "observations_total": count * 2,
            "observations_accepted": count * 2,
            "persistent_object_count": count,
        },
        "objects": objects,
    }


def test_d5_payload_normalizes_existing_semantics_without_measurement_authority() -> None:
    presentation = semantic_presentation_from_payload(_payload())

    assert presentation.session_id == "phase8-8d5-semantic-test"
    assert presentation.metric_geometry_status == "arbitrary"
    assert presentation.total_object_count == 3
    assert presentation.display_object_count == 3
    assert presentation.observations_accepted == 6
    assert presentation.certified_metrology is False
    assert "Semantic context" in presentation.authority_text
    assert "QC/measurement authority unchanged" in presentation.authority_text

    connector = presentation.objects[0]
    assert connector.class_name == "connector_body"
    assert connector.semantic_role == "connector_body"
    assert connector.observation_count == 2
    assert "Arbitrary reconstruction gauge" in connector.scale_text
    assert "not inspection/QC decision" in connector.authority_text
    assert "vertices" in connector.support_text
    assert "Semantic 93.0%" in connector.quality_text


def test_d5_presentation_is_bounded_but_preserves_total_object_count() -> None:
    payload = _payload(MAX_SEMANTIC_OBJECT_PRESENTATIONS + 37)
    presentation = semantic_presentation_from_payload(payload)

    assert presentation.total_object_count == MAX_SEMANTIC_OBJECT_PRESENTATIONS + 37
    assert presentation.display_object_count == MAX_SEMANTIC_OBJECT_PRESENTATIONS
    assert presentation.objects[0].object_id == "semantic-0000"
    assert presentation.objects[-1].object_id == f"semantic-{MAX_SEMANTIC_OBJECT_PRESENTATIONS - 1:04d}"


def test_d5_semantic_support_bounds_transform_to_bounded_viewer_segments() -> None:
    lower = np.asarray([-1.0, -2.0, 0.5], dtype=np.float64)
    upper = np.asarray([2.0, 3.0, 4.0], dtype=np.float64)
    segments = semantic_bounds_segments_in_viewer(lower, upper)
    center = semantic_centroid_in_viewer(np.asarray([0.5, 0.5, 2.25], dtype=np.float64))

    assert segments.shape == (24, 3)
    assert segments.dtype == np.float32
    assert np.isfinite(segments).all()
    assert center.shape == (3,)
    assert np.isfinite(center).all()


def test_d5_rejects_any_certified_metrology_claim() -> None:
    payload = _payload()
    payload["certified_metrology"] = True
    with pytest.raises(ValueError, match="certified metrology"):
        semantic_presentation_from_payload(payload)


@dataclass
class _Kind:
    value: str = "semantic_geometry"


class _Artifact:
    kind = _Kind()

    def __init__(self, *, snapshot: dict[str, object] | None, metadata: dict[str, object] | None = None):
        self._snapshot = snapshot or {}
        self._metadata = metadata or {}

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "semantic_geometry",
            "payload_snapshot": self._snapshot,
            "metadata": self._metadata,
        }


class _Scene:
    def __init__(self, session_id: str, artifact: _Artifact):
        self.session_id = session_id
        self.artifacts = (artifact,)


def test_d5_persistent_scene_uses_detailed_semantic_snapshot_when_present() -> None:
    payload = _payload()
    scene = _Scene(str(payload["session_id"]), _Artifact(snapshot=payload))

    presentation = semantic_presentation_from_persistent_scene(scene)

    assert presentation is not None
    assert presentation.detailed_snapshot is True
    assert presentation.display_object_count == 3
    assert presentation.source == "persistent_scene.semantic_geometry"


def test_d5_persistent_scene_metadata_only_remains_honest_and_does_not_invent_objects() -> None:
    scene = _Scene(
        "scene-summary-only",
        _Artifact(
            snapshot=None,
            metadata={
                "object_count": 17,
                "observations_total": 52,
                "observations_accepted": 49,
                "metric_geometry_status": "arbitrary",
            },
        ),
    )

    presentation = semantic_presentation_from_persistent_scene(scene)

    assert presentation is not None
    assert presentation.detailed_snapshot is False
    assert presentation.total_object_count == 17
    assert presentation.objects == ()
    assert "detailed snapshot unavailable" in presentation.structure_text
    assert presentation.certified_metrology is False
