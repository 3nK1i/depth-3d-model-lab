"""Structured runtime failure classification and recovery guidance.

Failures cross the worker/UI process boundary, so the public data types in this
module remain immutable, primitive-only and spawn-picklable.  Classification is
intentionally conservative: a failure is assigned a specific category only
when its exception type or message provides a strong signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureKind(Enum):
    CUDA_OUT_OF_MEMORY = "cuda_out_of_memory"
    MODEL_NOT_FOUND = "model_not_found"
    DEPENDENCY_MISSING = "dependency_missing"
    NETWORK_UNAVAILABLE = "network_unavailable"
    CHECKPOINT_CORRUPTED = "checkpoint_corrupted"
    AUTHENTICATION_FAILED = "authentication_failed"
    WARMUP_FAILED = "warmup_failed"
    INVALID_CONFIGURATION = "invalid_configuration"
    WORKER_TIMEOUT = "worker_timeout"
    WORKER_CRASHED = "worker_crashed"
    UNKNOWN_MODEL_ERROR = "unknown_model_error"


class FailureStage(Enum):
    LOAD = "load"
    WARMUP = "warmup"
    INFERENCE = "inference"
    UNLOAD = "unload"
    WORKER = "worker"


@dataclass(frozen=True, slots=True)
class FailureInfo:
    """Serializable failure information used by the worker and UI controller."""

    kind: FailureKind
    stage: FailureStage
    message: str
    technical_details: str = ""
    suggested_action: str = ""
    retryable: bool = True
    requires_worker_restart: bool = False
    preserve_source: bool = True

    @property
    def label(self) -> str:
        return failure_label(self.kind)


_LABELS: dict[FailureKind, str] = {
    FailureKind.CUDA_OUT_OF_MEMORY: "OOM RECOVERED",
    FailureKind.MODEL_NOT_FOUND: "MODEL NOT FOUND",
    FailureKind.DEPENDENCY_MISSING: "DEPENDENCY MISSING",
    FailureKind.NETWORK_UNAVAILABLE: "OFFLINE",
    FailureKind.CHECKPOINT_CORRUPTED: "CACHE ERROR",
    FailureKind.AUTHENTICATION_FAILED: "AUTH ERROR",
    FailureKind.WARMUP_FAILED: "WARMUP FAILED",
    FailureKind.INVALID_CONFIGURATION: "CONFIG ERROR",
    FailureKind.WORKER_TIMEOUT: "WORKER TIMEOUT",
    FailureKind.WORKER_CRASHED: "WORKER ERROR",
    FailureKind.UNKNOWN_MODEL_ERROR: "MODEL ERROR",
}


def failure_label(kind: FailureKind) -> str:
    return _LABELS.get(kind, "MODEL ERROR")


def classify_failure(
    exc: BaseException,
    *,
    stage: FailureStage,
    model_id: str = "",
) -> FailureInfo:
    """Classify an exception into a stable user-facing recovery category."""

    technical = _technical_details(exc)
    text = technical.lower()
    type_name = type(exc).__name__.lower()

    if _looks_like_oom(type_name, text):
        return FailureInfo(
            FailureKind.CUDA_OUT_OF_MEMORY,
            stage,
            "The selected model ran out of GPU memory.",
            technical,
            "Choose a safer profile or Development Mock, then retry.",
            retryable=True,
            requires_worker_restart=True,
        )

    if isinstance(exc, (ImportError, ModuleNotFoundError)) or _contains_any(
        text,
        (
            "dependency is not installed",
            "package is not installed",
            "no module named",
            "requires moge",
            "requires pytorch",
        ),
    ):
        return FailureInfo(
            FailureKind.DEPENDENCY_MISSING,
            stage,
            "A required model dependency is not installed.",
            technical,
            "Install the model-specific requirements, then retry.",
            retryable=True,
        )

    if _contains_any(
        text,
        (
            "401 client error",
            "403 client error",
            "unauthorized",
            "forbidden",
            "authentication failed",
            "invalid user token",
            "token signature",
            "gated repo",
            "permission denied",
        ),
    ):
        return FailureInfo(
            FailureKind.AUTHENTICATION_FAILED,
            stage,
            "The checkpoint could not be accessed with the current credentials.",
            technical,
            "Use a valid token for private/gated models or anonymous access for public models.",
            retryable=True,
        )

    if _contains_any(
        text,
        (
            "repository not found",
            "entry not found",
            "404 client error",
            "model not found",
            "unknown model id",
            "does not exist",
            "no such file or directory",
        ),
    ):
        return FailureInfo(
            FailureKind.MODEL_NOT_FOUND,
            stage,
            "The requested model checkpoint could not be found.",
            technical,
            "Verify the model identifier or restore the missing cached checkpoint.",
            retryable=True,
        )

    if _looks_like_network_failure(exc, type_name, text):
        return FailureInfo(
            FailureKind.NETWORK_UNAVAILABLE,
            stage,
            "The model is not available locally and the network could not be reached.",
            technical,
            "Reconnect to the network or use an already cached checkpoint.",
            retryable=True,
        )

    if _contains_any(
        text,
        (
            "corrupt",
            "corrupted",
            "checksum",
            "unexpected eof",
            "invalid header",
            "pickle data was truncated",
            "failed to read archive",
            "safetensorerror",
            "size mismatch",
            "bad zip file",
        ),
    ) or isinstance(exc, (EOFError, ValueError)) and _contains_any(
        text,
        ("checkpoint", "weights", "archive", "pickle", "safetensor"),
    ):
        return FailureInfo(
            FailureKind.CHECKPOINT_CORRUPTED,
            stage,
            "The cached model checkpoint appears to be incomplete or corrupted.",
            technical,
            "Remove only the affected checkpoint cache entry and download it again.",
            retryable=True,
            requires_worker_restart=False,
        )

    if stage is FailureStage.WARMUP or _contains_any(text, ("warm-up failed", "warmup failed")):
        return FailureInfo(
            FailureKind.WARMUP_FAILED,
            FailureStage.WARMUP,
            "The model loaded, but its warm-up inference failed.",
            technical,
            "Try a safer profile or restart the worker before retrying.",
            retryable=True,
            requires_worker_restart=_contains_any(text, ("cuda", "driver", "device")),
        )

    if isinstance(exc, (TypeError, ValueError)) or _contains_any(
        text,
        (
            "unsupported precision",
            "invalid configuration",
            "cuda device",
            "requested, but pytorch cannot access cuda",
            "bf16 is enabled only",
            "expects an h×w×3",
        ),
    ):
        return FailureInfo(
            FailureKind.INVALID_CONFIGURATION,
            stage,
            "The selected runtime configuration is not supported.",
            technical,
            "Choose a compatible device, precision, profile, or input configuration.",
            retryable=True,
        )

    model_context = f" for {model_id}" if model_id else ""
    return FailureInfo(
        FailureKind.UNKNOWN_MODEL_ERROR,
        stage,
        f"The model operation{model_context} failed unexpectedly.",
        technical,
        "Review the technical details, then retry or switch to Development Mock.",
        retryable=True,
        requires_worker_restart=False,
    )


def timeout_failure(command_name: str) -> FailureInfo:
    return FailureInfo(
        FailureKind.WORKER_TIMEOUT,
        FailureStage.WORKER,
        f"The inference worker did not acknowledge {command_name} in time.",
        command_name,
        "The worker will be restarted while the connected source is preserved.",
        retryable=True,
        requires_worker_restart=True,
    )


def crashed_worker_failure(exit_code: int | None = None) -> FailureInfo:
    details = ""
    if exit_code is not None:
        details = f"worker exit code: {exit_code}"
    return FailureInfo(
        FailureKind.WORKER_CRASHED,
        FailureStage.WORKER,
        "The inference worker process stopped unexpectedly.",
        details,
        "The worker will be restarted while the connected source is preserved.",
        retryable=True,
        requires_worker_restart=True,
    )


def format_failure_tooltip(failure: FailureInfo) -> str:
    parts = [failure.message]
    if failure.suggested_action:
        parts.append(f"Action: {failure.suggested_action}")
    if failure.technical_details:
        parts.append(f"Details: {failure.technical_details}")
    return "\n".join(parts)


def _technical_details(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _looks_like_oom(type_name: str, text: str) -> bool:
    return "outofmemory" in type_name or _contains_any(
        text,
        (
            "cuda out of memory",
            "out of gpu memory",
            "ran out of gpu memory",
            "simulated cuda out-of-memory",
            "cublas_status_alloc_failed",
        ),
    )


def _looks_like_network_failure(
    exc: BaseException,
    type_name: str,
    text: str,
) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if _contains_any(type_name, ("connectionerror", "connecttimeout", "readtimeout")):
        return True
    return _contains_any(
        text,
        (
            "network is unreachable",
            "connection refused",
            "connection aborted",
            "connection reset",
            "connection timed out",
            "temporary failure in name resolution",
            "name or service not known",
            "failed to establish a new connection",
            "max retries exceeded",
            "offline mode",
            "local_files_only",
            "cannot find the requested files in the disk cache",
        ),
    )
