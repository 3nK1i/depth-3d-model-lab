"""Session/frame ID tracking for stale-result rejection (spec §13).

Every frame submitted for inference and every result returned carries a
``session_id``. Stopping inference increments the session id
immediately; anything that arrives afterward tagged with the old id is
provably stale and gets dropped — this is what lets the worker finish
an in-flight CUDA forward pass naturally on stop, instead of needing to
be killed, without a late result corrupting the UI.
"""

from __future__ import annotations

import itertools
import threading


class SessionTracker:
    """Thread-safe session/frame id counters.

    Thread-safe because the UI thread (which calls ``stop`` to
    invalidate a session) and the process-manager's result-polling code
    both read/write the active session id.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session_counter = itertools.count(start=1)
        self._active_session_id = 0
        self._frame_counter = itertools.count(start=0)

    def start_new_session(self) -> int:
        """Increment and return the new active session id."""
        with self._lock:
            self._active_session_id = next(self._session_counter)
            self._frame_counter = itertools.count(start=0)
            return self._active_session_id

    def invalidate(self) -> None:
        """Invalidate the current session without starting a new one
        (e.g. on Stop, before returning to SOURCE_READY/MODEL_READY)."""
        with self._lock:
            self._active_session_id = 0

    def next_frame_id(self) -> int:
        with self._lock:
            return next(self._frame_counter)

    @property
    def active_session_id(self) -> int:
        with self._lock:
            return self._active_session_id

    def is_current(self, session_id: int, model_id: str | None = None, active_model_id: str | None = None) -> bool:
        """Whether a result tagged with ``session_id`` (and optionally
        ``model_id``) still belongs to the active session — spec §13:
        "ignore results when result.session_id != active_session_id or
        result.model_id != active_model_id"."""
        with self._lock:
            if session_id != self._active_session_id:
                return False
        if model_id is not None and active_model_id is not None and model_id != active_model_id:
            return False
        return True