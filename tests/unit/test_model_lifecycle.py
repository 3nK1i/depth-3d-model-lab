"""Explicit model lifecycle transition tests for Phase 5 Step 4.1."""

from __future__ import annotations

import pytest

from core.runtime.model_lifecycle import (
    InvalidModelLifecycleTransition,
    ModelLifecycle,
    ModelLifecycleState,
)


def test_complete_load_ready_unload_cycle() -> None:
    lifecycle = ModelLifecycle()

    for state in (
        ModelLifecycleState.LOADING,
        ModelLifecycleState.WARMING_UP,
        ModelLifecycleState.READY,
        ModelLifecycleState.UNLOADING,
        ModelLifecycleState.UNLOADED,
    ):
        assert lifecycle.transition_to(state)


def test_failure_can_retry_or_reset() -> None:
    lifecycle = ModelLifecycle()
    lifecycle.transition_to(ModelLifecycleState.LOADING)
    lifecycle.mark_failed()
    assert lifecycle.state is ModelLifecycleState.FAILED
    assert lifecycle.transition_to(ModelLifecycleState.LOADING)
    lifecycle.mark_failed()
    lifecycle.reset()
    assert lifecycle.state is ModelLifecycleState.UNLOADED


def test_invalid_transition_is_rejected() -> None:
    lifecycle = ModelLifecycle()
    with pytest.raises(InvalidModelLifecycleTransition):
        lifecycle.transition_to(ModelLifecycleState.READY)
    assert lifecycle.state is ModelLifecycleState.UNLOADED


def test_duplicate_stage_event_is_idempotent() -> None:
    lifecycle = ModelLifecycle()
    assert lifecycle.transition_to(ModelLifecycleState.LOADING)
    assert lifecycle.transition_to(ModelLifecycleState.LOADING)


def test_model_config_signature_ignores_render_only_point_budget() -> None:
    from core.runtime.model_lifecycle import model_config_signature

    first = {
        "device": "cuda:0",
        "precision": "fp16",
        "pretrained": "model-a",
        "point_budget": 250_000,
    }
    second = dict(first, point_budget=1_000_000)
    third = dict(first, pretrained="model-b")

    assert model_config_signature(first) == model_config_signature(second)
    assert model_config_signature(first) != model_config_signature(third)
