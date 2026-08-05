"""Authoritative validation entry point for tracking contracts."""

from __future__ import annotations

from core.tracking.contracts import ObjectTrackState


def validate_track_state(track: ObjectTrackState) -> ObjectTrackState:
    if not isinstance(track, ObjectTrackState):
        raise TypeError("track must be an ObjectTrackState.")
    return track
