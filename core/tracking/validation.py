"""Authoritative validation entry points for tracking contracts."""
from __future__ import annotations
from core.tracking.contracts import ObjectTrackState


def validate_track_state(track: ObjectTrackState) -> ObjectTrackState:
    if not isinstance(track,ObjectTrackState): raise TypeError("track must be an ObjectTrackState.")
    return track


def validate_tracking_result(result):
    from core.tracking.temporal_contracts import TemporalTrackingBatchResult
    if not isinstance(result,TemporalTrackingBatchResult): raise TypeError("result must be a TemporalTrackingBatchResult.")
    return result
