"""Dependency-light observation association using masks and boxes."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from core.perception.result import DenseMaskPayload, ObjectObservation
from core.tracking.contracts import TrackAssociation
from core.tracking.policy import AssociationPolicy


@dataclass(slots=True)
class ObservationGeometry:
    observation_id: str
    class_id: int
    class_name: str
    box_xyxy: tuple[float, float, float, float] | None
    mask: np.ndarray | None
    image_diagonal: float


def observation_geometry(observation: ObjectObservation) -> ObservationGeometry:
    box = observation.detection.box.xyxy if observation.detection is not None else None
    selected = observation.selected_mask
    mask = None
    if selected is not None and isinstance(selected.payload, DenseMaskPayload):
        mask = np.asarray(selected.payload.data, dtype=bool)
        if box is None and np.any(mask):
            ys, xs = np.nonzero(mask)
            box = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    size = observation.frame_reference.source_size
    return ObservationGeometry(
        observation.observation_id,
        observation.class_id,
        observation.class_name,
        box,
        mask,
        math.hypot(size.width, size.height),
    )


def box_iou(a, b) -> float | None:
    if a is None or b is None:
        return None
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def mask_iou(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or a.shape != b.shape:
        return None
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def centre_similarity(a, b, diagonal: float) -> float | None:
    if a is None or b is None or diagonal <= 0.0:
        return None
    ca = ((a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5)
    cb = ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)
    distance = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
    return float(max(0.0, 1.0 - distance / diagonal))


def association_score(previous: ObservationGeometry, current: ObservationGeometry, policy: AssociationPolicy) -> TrackAssociation | None:
    if previous.class_id != current.class_id:
        return None
    miou = mask_iou(previous.mask, current.mask)
    biou = box_iou(previous.box_xyxy, current.box_xyxy)
    centre = centre_similarity(previous.box_xyxy, current.box_xyxy, max(previous.image_diagonal, current.image_diagonal))
    metrics = (
        (miou, policy.mask_iou_weight),
        (biou, policy.box_iou_weight),
        (centre, policy.centre_similarity_weight),
    )
    denominator = sum(weight for value, weight in metrics if value is not None and weight > 0.0)
    if denominator <= 0.0:
        return None
    score = sum(float(value) * weight for value, weight in metrics if value is not None) / denominator
    if score < policy.minimum_score:
        return None
    return TrackAssociation("pending", current.observation_id, score, miou, biou, centre)


def greedy_associate(track_geometries: dict[str, ObservationGeometry], observations: tuple[ObjectObservation, ...], policy: AssociationPolicy):
    current = {obs.observation_id: observation_geometry(obs) for obs in observations}
    candidates=[]
    for track_id, previous in track_geometries.items():
        for observation_id, geometry in current.items():
            score = association_score(previous, geometry, policy)
            if score is not None:
                candidates.append((score.score, track_id, observation_id, score))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_tracks=set(); used_observations=set(); matched=[]
    for _, track_id, observation_id, score in candidates:
        if track_id in used_tracks or observation_id in used_observations:
            continue
        used_tracks.add(track_id); used_observations.add(observation_id)
        matched.append(TrackAssociation(track_id, observation_id, score.score, score.mask_iou, score.box_iou, score.centre_similarity))
    return tuple(matched), current
