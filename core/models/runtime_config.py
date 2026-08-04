"""Registry-selected runtime configuration builders.

Controllers provide a generic :class:`ModelRuntimeContext`; the selected model
descriptor decides which builder is called. This keeps model-specific choices
out of the application controller and inference worker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from core.models.metadata import ModelDescriptor, ModelProfileMetadata


@dataclass(frozen=True, slots=True)
class ModelRuntimeContext:
    """UI-side inputs used to resolve a primitive worker configuration."""

    hardware_info: Any = None
    resolved_profile: Any = None
    point_budget: int | None = None
    overrides: Mapping[str, object] = field(default_factory=dict)


def build_profile_runtime_config(
    descriptor: ModelDescriptor,
    context: ModelRuntimeContext,
) -> dict[str, object]:
    """Build config for models using the Phase 5 hardware profile resolver."""

    resolved = context.resolved_profile
    if resolved is None or not hasattr(resolved, "as_model_config"):
        raise ValueError(
            f"Model {descriptor.model_id!r} requires a resolved compute profile."
        )
    config = dict(descriptor.default_config)
    config.update(dict(resolved.as_model_config()))
    if context.point_budget is not None:
        config["point_budget"] = max(1, int(context.point_budget))
    config.update(dict(context.overrides))
    return validate_runtime_config(config)


def build_static_runtime_config(
    descriptor: ModelDescriptor,
    context: ModelRuntimeContext,
) -> dict[str, object]:
    """Build config for adapters with registry-declared fixed defaults."""

    config = dict(descriptor.default_config)
    config.update(dict(context.overrides))
    return validate_runtime_config(config)


def validate_runtime_config(config: Mapping[str, object]) -> dict[str, object]:
    """Return a picklable primitive config or reject the invalid value early."""

    result: dict[str, object] = {}
    for key, value in config.items():
        name = str(key)
        if value is None or isinstance(value, (str, int, float, bool)):
            result[name] = value
            continue
        if isinstance(value, (tuple, list)) and all(
            item is None or isinstance(item, (str, int, float, bool)) for item in value
        ):
            result[name] = list(value)
            continue
        raise TypeError(
            f"Runtime config value {name!r} for model registry is not primitive: "
            f"{type(value).__name__}."
        )
    return result


def build_profile_metadata(
    descriptor: ModelDescriptor,
    resolved_profile: Any,
) -> ModelProfileMetadata:
    """Expose the existing hardware profile through registry-owned UI data."""

    if resolved_profile is None or not hasattr(resolved_profile, "spec"):
        raise ValueError(
            f"Model {descriptor.model_id!r} requires a resolved profile for presentation."
        )
    spec = resolved_profile.spec
    precision = getattr(resolved_profile.precision, "display_name", str(resolved_profile.precision))
    detail = f"Level {int(spec.resolution_level)} · {int(spec.num_tokens)} tok"
    tooltip = (
        "Auto tuning: "
        f"{spec.model_label}, {precision}, level {int(spec.resolution_level)}, "
        f"{int(spec.num_tokens)} tokens, {int(spec.warmup_runs)} warm-up run(s)."
    )
    warning = getattr(resolved_profile, "warning", None)
    if warning:
        tooltip += f"\nProfile: {warning}"
    return ModelProfileMetadata(
        variant_label=str(spec.model_label),
        detail_label=detail,
        precision_label=str(precision),
        point_budget=max(1, int(spec.point_budget)),
        live_point_budget=max(1, int(spec.live_point_budget)),
        tooltip=tooltip,
    )


def build_mock_profile_metadata(
    descriptor: ModelDescriptor,
    resolved_profile: Any,
) -> ModelProfileMetadata:
    """Present the mock without pretending it has an AI checkpoint/profile."""

    spec = getattr(resolved_profile, "spec", None)
    point_budget = max(1, int(getattr(spec, "point_budget", 250_000)))
    live_budget = max(1, int(getattr(spec, "live_point_budget", 100_000)))
    return ModelProfileMetadata(
        variant_label="Synthetic Diagnostic",
        detail_label="No AI inference",
        precision_label="FP32",
        point_budget=point_budget,
        live_point_budget=live_budget,
        tooltip=descriptor.presentation.description,
    )
