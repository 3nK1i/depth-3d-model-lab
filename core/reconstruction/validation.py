"""Deterministic validation helpers for reconstruction contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Iterable
from uuid import UUID

import numpy as np

from core.contracts.validation import require_finite, require_nonempty
from core.reconstruction.errors import ReconstructionValidationError


def require_uuid(value: object, *, field_name: str) -> str:
    """Return one canonical lowercase UUID string."""

    text = require_nonempty(value, field_name=field_name)
    try:
        parsed = UUID(text)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ReconstructionValidationError(f"{field_name} must be a valid UUID.") from exc
    return str(parsed)


def require_nonnegative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ReconstructionValidationError(f"{field_name} must be a non-negative integer.")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ReconstructionValidationError(
            f"{field_name} must be a non-negative integer."
        ) from exc
    if number < 0 or number != value:
        raise ReconstructionValidationError(f"{field_name} must be a non-negative integer.")
    return number


def require_positive_int(value: object, *, field_name: str) -> int:
    number = require_nonnegative_int(value, field_name=field_name)
    if number <= 0:
        raise ReconstructionValidationError(f"{field_name} must be a positive integer.")
    return number


def require_timestamp_ns(value: object, *, field_name: str = "timestamp_ns") -> int:
    return require_nonnegative_int(value, field_name=field_name)


def require_probability(value: object, *, field_name: str) -> float:
    number = require_finite(value, field_name=field_name)
    if number < 0.0 or number > 1.0:
        raise ReconstructionValidationError(f"{field_name} must be within [0, 1].")
    return number


def require_safe_relative_path(value: object, *, field_name: str) -> str:
    path = PurePosixPath(require_nonempty(value, field_name=field_name))
    if path.is_absolute() or ".." in path.parts:
        raise ReconstructionValidationError(
            f"{field_name} must be a safe relative POSIX path without traversal."
        )
    return path.as_posix()


def require_sha256(value: object, *, field_name: str = "sha256", allow_empty: bool = True) -> str:
    digest = str(value).strip().lower()
    if not digest and allow_empty:
        return ""
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ReconstructionValidationError(
            f"{field_name} must be a 64-character hexadecimal digest."
        )
    return digest


def require_iso8601_utc(value: object, *, field_name: str) -> str:
    text = require_nonempty(value, field_name=field_name)
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ReconstructionValidationError(f"{field_name} must be ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ReconstructionValidationError(f"{field_name} must include the UTC timezone.")
    return parsed.astimezone(timezone.utc).isoformat()


def require_strictly_increasing(values: Iterable[int], *, field_name: str) -> tuple[int, ...]:
    normalized = tuple(require_timestamp_ns(value, field_name=field_name) for value in values)
    if any(current <= previous for previous, current in zip(normalized, normalized[1:])):
        raise ReconstructionValidationError(f"{field_name} values must be strictly increasing.")
    return normalized


def readonly_vector3(value: object, *, field_name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ReconstructionValidationError(f"{field_name} must be a finite 3-vector.")
    vector = np.array(vector, copy=True)
    vector.setflags(write=False)
    return vector
