"""Structured failure classification and recovery-guidance tests."""

from __future__ import annotations

import pickle

import pytest

from core.runtime.failures import (
    FailureInfo,
    FailureKind,
    FailureStage,
    classify_failure,
    crashed_worker_failure,
    failure_label,
    format_failure_tooltip,
    timeout_failure,
)


@pytest.mark.parametrize(
    ("exc", "stage", "expected"),
    [
        (
            RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
            FailureStage.INFERENCE,
            FailureKind.CUDA_OUT_OF_MEMORY,
        ),
        (
            ModuleNotFoundError("No module named 'moge'"),
            FailureStage.LOAD,
            FailureKind.DEPENDENCY_MISSING,
        ),
        (
            RuntimeError("401 Client Error: Unauthorized; invalid user token"),
            FailureStage.LOAD,
            FailureKind.AUTHENTICATION_FAILED,
        ),
        (
            RuntimeError("Repository Not Found: 404 Client Error"),
            FailureStage.LOAD,
            FailureKind.MODEL_NOT_FOUND,
        ),
        (
            ConnectionError("Network is unreachable"),
            FailureStage.LOAD,
            FailureKind.NETWORK_UNAVAILABLE,
        ),
        (
            RuntimeError("safetensorerror: invalid header / corrupted checkpoint"),
            FailureStage.LOAD,
            FailureKind.CHECKPOINT_CORRUPTED,
        ),
        (
            RuntimeError("MoGe-2 warm-up failed: kernel launch failed"),
            FailureStage.WARMUP,
            FailureKind.WARMUP_FAILED,
        ),
        (
            ValueError("Unsupported precision: int8"),
            FailureStage.LOAD,
            FailureKind.INVALID_CONFIGURATION,
        ),
        (
            RuntimeError("something unexpected happened"),
            FailureStage.INFERENCE,
            FailureKind.UNKNOWN_MODEL_ERROR,
        ),
    ],
)
def test_classify_failure_categories(exc, stage, expected) -> None:
    failure = classify_failure(exc, stage=stage, model_id="moge-2")
    assert failure.kind is expected
    assert failure.stage is stage or expected is FailureKind.WARMUP_FAILED
    assert failure.message
    assert failure.suggested_action
    assert failure.preserve_source


def test_oom_requires_worker_restart_and_is_retryable() -> None:
    failure = classify_failure(
        RuntimeError("simulated CUDA out-of-memory (mock)"),
        stage=FailureStage.INFERENCE,
    )
    assert failure.kind is FailureKind.CUDA_OUT_OF_MEMORY
    assert failure.requires_worker_restart
    assert failure.retryable
    assert failure.label == "OOM RECOVERED"


def test_timeout_and_crash_helpers_are_restartable() -> None:
    timeout = timeout_failure("load_model")
    crash = crashed_worker_failure(-9)

    assert timeout.kind is FailureKind.WORKER_TIMEOUT
    assert timeout.requires_worker_restart
    assert crash.kind is FailureKind.WORKER_CRASHED
    assert "-9" in crash.technical_details


def test_failure_info_is_spawn_picklable_and_immutable() -> None:
    failure = FailureInfo(
        FailureKind.NETWORK_UNAVAILABLE,
        FailureStage.LOAD,
        "offline",
        "ConnectionError: offline",
        "Reconnect",
    )
    restored = pickle.loads(pickle.dumps(failure))
    assert restored == failure
    with pytest.raises((AttributeError, TypeError)):
        failure.message = "changed"  # type: ignore[misc]


def test_tooltip_contains_action_and_technical_details() -> None:
    failure = timeout_failure("stop")
    tooltip = format_failure_tooltip(failure)
    assert failure.message in tooltip
    assert "Action:" in tooltip
    assert "Details:" in tooltip
    assert failure_label(failure.kind) == "WORKER TIMEOUT"
