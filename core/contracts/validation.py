"""Dependency-light validation helpers for Phase 7.5 contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from math import isfinite
from typing import Any, TypeAlias

import numpy as np


class ContractValidationError(ValueError):
    """Raised when a public data contract contains inconsistent semantics."""


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | tuple[tuple[str, "JsonValue"], ...]
MetadataItems: TypeAlias = tuple[tuple[str, JsonValue], ...]


def require_nonempty(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ContractValidationError(f"{field_name} must not be empty.")
    return text


def require_finite(value: object, *, field_name: str) -> float:
    number = float(value)
    if not isfinite(number):
        raise ContractValidationError(f"{field_name} must be finite.")
    return number


def require_probability(value: object, *, field_name: str) -> float:
    number = require_finite(value, field_name=field_name)
    if number < 0.0 or number > 1.0:
        raise ContractValidationError(f"{field_name} must be within [0, 1].")
    return number


def readonly_array(
    value: object,
    *,
    field_name: str,
    dtype: np.dtype[Any] | type | None = None,
    ndim: int | None = None,
    shape: tuple[int | None, ...] | None = None,
    finite: bool = False,
    copy: bool = True,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if copy:
        array = np.array(array, copy=True)
    if ndim is not None and array.ndim != ndim:
        raise ContractValidationError(
            f"{field_name} must have {ndim} dimensions; received {array.ndim}."
        )
    if shape is not None:
        if array.ndim != len(shape):
            raise ContractValidationError(
                f"{field_name} must have shape {shape}; received {array.shape}."
            )
        for actual, expected in zip(array.shape, shape, strict=True):
            if expected is not None and actual != expected:
                raise ContractValidationError(
                    f"{field_name} must have shape {shape}; received {array.shape}."
                )
    if finite and not np.isfinite(array).all():
        raise ContractValidationError(f"{field_name} must contain only finite values.")
    array.setflags(write=False)
    return array


def normalize_metadata(value: Mapping[str, Any] | Sequence[tuple[str, Any]] | None) -> MetadataItems:
    """Return deterministic, pickle-safe, JSON-compatible metadata items."""

    if value is None:
        return ()
    items = value.items() if isinstance(value, Mapping) else value
    normalized: list[tuple[str, JsonValue]] = []
    seen: set[str] = set()
    for raw_key, raw_value in items:
        key = require_nonempty(raw_key, field_name="metadata key")
        if key in seen:
            raise ContractValidationError(f"Duplicate metadata key: {key!r}.")
        seen.add(key)
        normalized.append((key, normalize_json_value(raw_value, field_name=f"metadata[{key!r}]")))
    normalized.sort(key=lambda item: item[0])
    return tuple(normalized)


def normalize_json_value(value: Any, *, field_name: str = "value") -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return require_finite(value, field_name=field_name)
    if isinstance(value, Enum):
        return normalize_json_value(value.value, field_name=field_name)
    if isinstance(value, np.generic):
        return normalize_json_value(value.item(), field_name=field_name)
    if isinstance(value, Mapping):
        return normalize_metadata(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            normalize_json_value(item, field_name=f"{field_name}[]") for item in value
        )
    raise ContractValidationError(
        f"{field_name} must be JSON-compatible; received {type(value).__name__}."
    )


def metadata_as_dict(items: MetadataItems) -> dict[str, Any]:
    return {key: json_value_as_python(value) for key, value in items}


def json_value_as_python(value: JsonValue) -> Any:
    if isinstance(value, tuple):
        if all(
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            for item in value
        ):
            return {str(key): json_value_as_python(item) for key, item in value}  # type: ignore[misc]
        return [json_value_as_python(item) for item in value]
    return value
