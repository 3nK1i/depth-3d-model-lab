"""Latest-frame-only mailbox for the live SLAM control bridge."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from core.reconstruction.live_contracts import LiveSlamFrame


@dataclass(frozen=True, slots=True)
class LiveFrameMailboxStats:
    offered_frames: int
    delivered_frames: int
    replaced_frames: int
    pending: bool


class LatestLiveFrameMailbox:
    """Retain at most one frame envelope so control IPC can never build a backlog."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._pending: LiveSlamFrame | None = None
        self._offered = 0
        self._delivered = 0
        self._replaced = 0

    def offer(self, frame: LiveSlamFrame) -> bool:
        if not isinstance(frame, LiveSlamFrame):
            raise TypeError("frame must be LiveSlamFrame.")
        with self._lock:
            replaced = self._pending is not None
            if replaced:
                self._replaced += 1
            self._pending = frame
            self._offered += 1
            return replaced

    def take_latest(self) -> LiveSlamFrame | None:
        with self._lock:
            frame = self._pending
            self._pending = None
            if frame is not None:
                self._delivered += 1
            return frame

    def clear(self) -> None:
        with self._lock:
            self._pending = None

    @property
    def stats(self) -> LiveFrameMailboxStats:
        with self._lock:
            return LiveFrameMailboxStats(
                offered_frames=self._offered,
                delivered_frames=self._delivered,
                replaced_frames=self._replaced,
                pending=self._pending is not None,
            )
