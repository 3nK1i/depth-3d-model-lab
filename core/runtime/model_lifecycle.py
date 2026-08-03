"""Explicit model lifecycle state used by the UI controller.

The application state machine describes source/runtime behaviour.  This module
tracks the independent lifetime of the model that lives inside the spawned
worker process so load, warm-up, ready, unload, failure and retry paths remain
unambiguous.
"""

from __future__ import annotations

from enum import Enum


class ModelLifecycleState(Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    WARMING_UP = "warming_up"
    READY = "ready"
    UNLOADING = "unloading"
    FAILED = "failed"


_ALLOWED_TRANSITIONS: dict[ModelLifecycleState, frozenset[ModelLifecycleState]] = {
    ModelLifecycleState.UNLOADED: frozenset(
        {ModelLifecycleState.LOADING, ModelLifecycleState.FAILED}
    ),
    ModelLifecycleState.LOADING: frozenset(
        {
            ModelLifecycleState.WARMING_UP,
            ModelLifecycleState.READY,
            ModelLifecycleState.UNLOADED,
            ModelLifecycleState.FAILED,
        }
    ),
    ModelLifecycleState.WARMING_UP: frozenset(
        {
            ModelLifecycleState.READY,
            ModelLifecycleState.UNLOADED,
            ModelLifecycleState.FAILED,
        }
    ),
    ModelLifecycleState.READY: frozenset(
        {ModelLifecycleState.UNLOADING, ModelLifecycleState.FAILED}
    ),
    ModelLifecycleState.UNLOADING: frozenset(
        {ModelLifecycleState.UNLOADED, ModelLifecycleState.FAILED}
    ),
    ModelLifecycleState.FAILED: frozenset(
        {ModelLifecycleState.UNLOADED, ModelLifecycleState.LOADING}
    ),
}


class InvalidModelLifecycleTransition(RuntimeError):
    """Raised when model lifecycle code attempts an impossible transition."""


class ModelLifecycle:
    """Small pure-Python tracker suitable for production and unit tests."""

    def __init__(self) -> None:
        self._state = ModelLifecycleState.UNLOADED

    @property
    def state(self) -> ModelLifecycleState:
        return self._state

    def can_transition_to(self, target: ModelLifecycleState) -> bool:
        if target is self._state:
            return True
        return target in _ALLOWED_TRANSITIONS.get(self._state, frozenset())

    def transition_to(
        self,
        target: ModelLifecycleState,
        *,
        strict: bool = True,
    ) -> bool:
        if target is self._state:
            return True
        if not self.can_transition_to(target):
            if strict:
                raise InvalidModelLifecycleTransition(
                    f"Cannot transition model lifecycle from {self._state} to {target}"
                )
            return False
        self._state = target
        return True

    def mark_failed(self) -> None:
        self._state = ModelLifecycleState.FAILED

    def reset(self) -> None:
        self._state = ModelLifecycleState.UNLOADED


_NON_IDENTITY_CONFIG_KEYS = frozenset({"point_budget"})


def model_config_signature(config: dict[str, object]) -> tuple:
    """Return a stable signature for settings that require a model reload.

    Render-only/per-frame settings such as ``point_budget`` are deliberately
    excluded because changing them must not unload and reload model weights.
    """

    def freeze(value: object):
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (str(key), freeze(item))
                    for key, item in value.items()
                    if str(key) not in _NON_IDENTITY_CONFIG_KEYS
                )
            )
        if isinstance(value, (list, tuple)):
            return tuple(freeze(item) for item in value)
        if isinstance(value, set):
            return tuple(sorted(freeze(item) for item in value))
        return value

    return freeze(config)
