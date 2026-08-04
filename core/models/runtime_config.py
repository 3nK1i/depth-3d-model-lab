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



def build_depth_pro_runtime_config(
    descriptor: ModelDescriptor,
    context: ModelRuntimeContext,
) -> dict[str, object]:
    """Build a Depth Pro config without leaking model logic into controllers.

    Depth Pro currently supports FP32 and FP16. The shared profile resolver may
    expose BF16 for future hardware, so this builder safely maps that case to
    FP16 on CUDA while retaining FP32 on CPU. Profile tiers control precision,
    point-cloud density, warm-up repetitions, and backend tuning; the official
    model architecture and its internal 1536-pixel inference size remain fixed.
    """

    resolved = context.resolved_profile
    if resolved is None or not hasattr(resolved, "spec"):
        raise ValueError(
            f"Model {descriptor.model_id!r} requires a resolved compute profile."
        )

    spec = resolved.spec
    device = str(getattr(resolved, "device", "cpu"))
    precision_value = getattr(getattr(resolved, "precision", None), "value", "fp32")
    precision = str(precision_value).lower()
    if device.startswith("cpu"):
        precision = "fp32"
    elif precision not in {"fp32", "fp16"}:
        precision = "fp16"

    effective = getattr(getattr(resolved, "effective", None), "value", "efficient")
    config = dict(descriptor.default_config)
    config.update(
        {
            "profile": str(effective),
            "device": device,
            "precision": precision,
            "point_budget": max(1, int(spec.point_budget)),
            "live_point_budget": max(1, int(spec.live_point_budget)),
            "warmup_size": max(64, int(spec.warmup_size)),
            "warmup_runs": max(1, int(spec.warmup_runs)),
            "allow_tf32": bool(spec.allow_tf32),
            "matmul_precision": str(spec.matmul_precision),
            "hardware_tier": str(effective),
        }
    )
    if context.point_budget is not None:
        config["point_budget"] = max(1, int(context.point_budget))
    config.update(dict(context.overrides))
    return validate_runtime_config(config)


def build_depth_pro_profile_metadata(
    descriptor: ModelDescriptor,
    resolved_profile: Any,
) -> ModelProfileMetadata:
    """Expose Depth Pro's fixed architecture through generic profile widgets."""

    if resolved_profile is None or not hasattr(resolved_profile, "spec"):
        raise ValueError(
            f"Model {descriptor.model_id!r} requires a resolved profile for presentation."
        )
    spec = resolved_profile.spec
    device = str(getattr(resolved_profile, "device", "cpu"))
    precision_value = getattr(getattr(resolved_profile, "precision", None), "value", "fp32")
    precision = str(precision_value).lower()
    if device.startswith("cpu"):
        precision = "fp32"
    elif precision not in {"fp32", "fp16"}:
        precision = "fp16"
    precision_label = "FP16" if precision == "fp16" else "FP32"

    tooltip = (
        "Depth Pro uses the official DINOv2-L architecture and a fixed 1536×1536 "
        "internal inference size. The selected hardware profile controls precision, "
        "warm-up, and point-cloud density."
    )
    warning = getattr(resolved_profile, "warning", None)
    if warning:
        tooltip += f"\nProfile: {warning}"
    return ModelProfileMetadata(
        variant_label="Depth Pro · DINOv2-L",
        detail_label="1536² internal · Metric",
        precision_label=precision_label,
        point_budget=max(1, int(spec.point_budget)),
        live_point_budget=max(1, int(spec.live_point_budget)),
        tooltip=tooltip,
    )

_UNIDEPTH_V2_PROFILE_VARIANTS: dict[str, dict[str, object]] = {
    "efficient": {
        "variant": "vits14",
        "repository": "lpiccinelli/unidepth-v2-vits14",
        "checkpoint_dir": "checkpoints/unidepth-v2/vits14",
        "model_label": "UniDepthV2 ViT-S/14",
        "resolution_level": 2,
    },
    "balanced": {
        "variant": "vitb14",
        "repository": "lpiccinelli/unidepth-v2-vitb14",
        "checkpoint_dir": "checkpoints/unidepth-v2/vitb14",
        "model_label": "UniDepthV2 ViT-B/14",
        "resolution_level": 5,
    },
    "ultra": {
        "variant": "vitl14",
        "repository": "lpiccinelli/unidepth-v2-vitl14",
        "checkpoint_dir": "checkpoints/unidepth-v2/vitl14",
        "model_label": "UniDepthV2 ViT-L/14",
        "resolution_level": 8,
    },
}


def _unidepth_v2_profile_values(resolved_profile: Any) -> tuple[str, dict[str, object]]:
    effective = str(getattr(getattr(resolved_profile, "effective", None), "value", "efficient"))
    if effective not in _UNIDEPTH_V2_PROFILE_VARIANTS:
        effective = "efficient"
    return effective, dict(_UNIDEPTH_V2_PROFILE_VARIANTS[effective])


def build_unidepth_v2_runtime_config(
    descriptor: ModelDescriptor,
    context: ModelRuntimeContext,
) -> dict[str, object]:
    """Resolve a GPU-safe UniDepthV2 variant entirely through the registry."""

    resolved = context.resolved_profile
    if resolved is None or not hasattr(resolved, "spec"):
        raise ValueError(
            f"Model {descriptor.model_id!r} requires a resolved compute profile."
        )
    spec = resolved.spec
    effective, variant = _unidepth_v2_profile_values(resolved)
    device = str(getattr(resolved, "device", "cpu"))
    precision_value = getattr(getattr(resolved, "precision", None), "value", "fp32")
    precision = str(precision_value).lower()
    if device.startswith("cpu"):
        precision = "fp32"
    elif precision not in {"fp32", "fp16"}:
        precision = "fp16"

    config = dict(descriptor.default_config)
    config.update(variant)
    config.update(
        {
            "profile": effective,
            "device": device,
            "precision": precision,
            "point_budget": max(1, int(spec.point_budget)),
            "live_point_budget": max(1, int(spec.live_point_budget)),
            "warmup_size": max(64, int(spec.warmup_size)),
            "warmup_runs": max(1, int(spec.warmup_runs)),
            "allow_tf32": bool(spec.allow_tf32),
            "matmul_precision": str(spec.matmul_precision),
            "hardware_tier": effective,
        }
    )
    if context.point_budget is not None:
        config["point_budget"] = max(1, int(context.point_budget))
    config.update(dict(context.overrides))
    return validate_runtime_config(config)


def build_unidepth_v2_profile_metadata(
    descriptor: ModelDescriptor,
    resolved_profile: Any,
) -> ModelProfileMetadata:
    """Expose registry-selected UniDepthV2 variants in generic settings widgets."""

    if resolved_profile is None or not hasattr(resolved_profile, "spec"):
        raise ValueError(
            f"Model {descriptor.model_id!r} requires a resolved profile for presentation."
        )
    spec = resolved_profile.spec
    _, variant = _unidepth_v2_profile_values(resolved_profile)
    device = str(getattr(resolved_profile, "device", "cpu"))
    precision_value = getattr(getattr(resolved_profile, "precision", None), "value", "fp32")
    precision = str(precision_value).lower()
    if device.startswith("cpu"):
        precision = "fp32"
    elif precision not in {"fp32", "fp16"}:
        precision = "fp16"
    precision_label = "FP16" if precision == "fp16" else "FP32"
    level = int(variant["resolution_level"])
    tooltip = (
        f"Automatic UniDepthV2 selection: {variant['model_label']}, "
        f"resolution level {level}, {precision_label}. Confidence is exposed as "
        "relative uncertainty; lower values are more reliable within the same frame."
    )
    warning = getattr(resolved_profile, "warning", None)
    if warning:
        tooltip += f"\nProfile: {warning}"
    return ModelProfileMetadata(
        variant_label=str(variant["model_label"]),
        detail_label=f"Level {level} · Metric + uncertainty",
        precision_label=precision_label,
        point_budget=max(1, int(spec.point_budget)),
        live_point_budget=max(1, int(spec.live_point_budget)),
        tooltip=tooltip,
    )


_METRIC3D_V2_PROFILE_VARIANTS: dict[str, dict[str, object]] = {
    "efficient": {
        "variant": "vit-small",
        "entrypoint": "metric3d_vit_small",
        "checkpoint": "checkpoints/metric3d-v2/metric_depth_vit_small_800k.pth",
        "model_label": "Metric3D v2 ViT-S",
        "decoder_label": "RAFT-4iter",
        "input_height": 616,
        "input_width": 1064,
    },
    "balanced": {
        "variant": "vit-small",
        "entrypoint": "metric3d_vit_small",
        "checkpoint": "checkpoints/metric3d-v2/metric_depth_vit_small_800k.pth",
        "model_label": "Metric3D v2 ViT-S",
        "decoder_label": "RAFT-4iter",
        "input_height": 616,
        "input_width": 1064,
    },
    "ultra": {
        "variant": "vit-large",
        "entrypoint": "metric3d_vit_large",
        "checkpoint": "checkpoints/metric3d-v2/metric_depth_vit_large_800k.pth",
        "model_label": "Metric3D v2 ViT-L",
        "decoder_label": "RAFT-8iter",
        "input_height": 616,
        "input_width": 1064,
    },
}


def _metric3d_v2_profile_values(resolved_profile: Any) -> tuple[str, dict[str, object]]:
    effective = str(
        getattr(getattr(resolved_profile, "effective", None), "value", "efficient")
    )
    if effective not in _METRIC3D_V2_PROFILE_VARIANTS:
        effective = "efficient"
    return effective, dict(_METRIC3D_V2_PROFILE_VARIANTS[effective])


def build_metric3d_v2_runtime_config(
    descriptor: ModelDescriptor,
    context: ModelRuntimeContext,
) -> dict[str, object]:
    """Resolve a local, GPU-safe Metric3D v2 configuration via the registry."""

    resolved = context.resolved_profile
    if resolved is None or not hasattr(resolved, "spec"):
        raise ValueError(
            f"Model {descriptor.model_id!r} requires a resolved compute profile."
        )
    spec = resolved.spec
    effective, variant = _metric3d_v2_profile_values(resolved)
    device = str(getattr(resolved, "device", "cpu"))
    precision_value = getattr(getattr(resolved, "precision", None), "value", "fp32")
    precision = str(precision_value).lower()
    if device.startswith("cpu"):
        precision = "fp32"
    elif precision not in {"fp32", "fp16"}:
        precision = "fp16"

    config = dict(descriptor.default_config)
    config.update(variant)
    config.update(
        {
            "profile": effective,
            "device": device,
            "precision": precision,
            "point_budget": max(1, int(spec.point_budget)),
            "live_point_budget": max(1, int(spec.live_point_budget)),
            "warmup_size": max(64, int(spec.warmup_size)),
            "warmup_runs": max(1, int(spec.warmup_runs)),
            "allow_tf32": bool(spec.allow_tf32),
            "matmul_precision": str(spec.matmul_precision),
            "hardware_tier": effective,
        }
    )
    if context.point_budget is not None:
        config["point_budget"] = max(1, int(context.point_budget))
    config.update(dict(context.overrides))
    return validate_runtime_config(config)


def build_metric3d_v2_profile_metadata(
    descriptor: ModelDescriptor,
    resolved_profile: Any,
) -> ModelProfileMetadata:
    """Expose registry-selected Metric3D v2 variants to the generic UI."""

    if resolved_profile is None or not hasattr(resolved_profile, "spec"):
        raise ValueError(
            f"Model {descriptor.model_id!r} requires a resolved profile for presentation."
        )
    spec = resolved_profile.spec
    _, variant = _metric3d_v2_profile_values(resolved_profile)
    device = str(getattr(resolved_profile, "device", "cpu"))
    precision_value = getattr(getattr(resolved_profile, "precision", None), "value", "fp32")
    precision = str(precision_value).lower()
    if device.startswith("cpu"):
        precision = "fp32"
    elif precision not in {"fp32", "fp16"}:
        precision = "fp16"
    precision_label = "FP16" if precision == "fp16" else "FP32"
    tooltip = (
        f"Automatic Metric3D v2 selection: {variant['model_label']} with "
        f"{variant['decoder_label']}, {precision_label}. Metric scale depends on "
        "camera intrinsics; until calibration is available, the application uses "
        "estimated intrinsics and labels measurements as estimated."
    )
    warning = getattr(resolved_profile, "warning", None)
    if warning:
        tooltip += f"\nProfile: {warning}"
    return ModelProfileMetadata(
        variant_label=str(variant["model_label"]),
        detail_label=(
            f"{int(variant['input_width'])}×{int(variant['input_height'])} · "
            "Metric + normals"
        ),
        precision_label=precision_label,
        point_budget=max(1, int(spec.point_budget)),
        live_point_budget=max(1, int(spec.live_point_budget)),
        tooltip=tooltip,
    )



_DEPTH_ANYTHING_V2_PROFILE_VARIANTS: dict[str, dict[str, object]] = {
    "efficient": {
        "encoder": "vits",
        "checkpoint": "checkpoints/depth-anything-v2/depth_anything_v2_vits.pth",
        "model_label": "Depth Anything V2 Small",
        "input_size": 518,
    },
    "balanced": {
        "encoder": "vitb",
        "checkpoint": "checkpoints/depth-anything-v2/depth_anything_v2_vitb.pth",
        "model_label": "Depth Anything V2 Base",
        "input_size": 518,
    },
    "ultra": {
        "encoder": "vitl",
        "checkpoint": "checkpoints/depth-anything-v2/depth_anything_v2_vitl.pth",
        "model_label": "Depth Anything V2 Large",
        "input_size": 700,
    },
}


def _depth_anything_v2_profile_values(resolved_profile: Any) -> tuple[str, dict[str, object]]:
    effective = str(getattr(getattr(resolved_profile, "effective", None), "value", "efficient"))
    if effective not in _DEPTH_ANYTHING_V2_PROFILE_VARIANTS:
        effective = "efficient"
    return effective, dict(_DEPTH_ANYTHING_V2_PROFILE_VARIANTS[effective])


def build_depth_anything_v2_runtime_config(
    descriptor: ModelDescriptor,
    context: ModelRuntimeContext,
) -> dict[str, object]:
    """Resolve image Depth Anything V2 variants through the registry."""

    resolved = context.resolved_profile
    if resolved is None or not hasattr(resolved, "spec"):
        raise ValueError(
            f"Model {descriptor.model_id!r} requires a resolved compute profile."
        )
    spec = resolved.spec
    effective, variant = _depth_anything_v2_profile_values(resolved)
    device = str(getattr(resolved, "device", "cpu"))
    precision_value = getattr(getattr(resolved, "precision", None), "value", "fp32")
    precision = str(precision_value).lower()
    if device.startswith("cpu"):
        precision = "fp32"
    elif precision not in {"fp32", "fp16"}:
        precision = "fp16"
    config = dict(descriptor.default_config)
    config.update(variant)
    config.update(
        {
            "profile": effective,
            "device": device,
            "precision": precision,
            "point_budget": max(1, int(spec.point_budget)),
            "live_point_budget": max(1, int(spec.live_point_budget)),
            "warmup_size": max(64, int(spec.warmup_size)),
            "warmup_runs": max(1, int(spec.warmup_runs)),
            "allow_tf32": bool(spec.allow_tf32),
            "matmul_precision": str(spec.matmul_precision),
            "hardware_tier": effective,
        }
    )
    if context.point_budget is not None:
        config["point_budget"] = max(1, int(context.point_budget))
    config.update(dict(context.overrides))
    return validate_runtime_config(config)


def build_depth_anything_v2_profile_metadata(
    descriptor: ModelDescriptor,
    resolved_profile: Any,
) -> ModelProfileMetadata:
    if resolved_profile is None or not hasattr(resolved_profile, "spec"):
        raise ValueError(
            f"Model {descriptor.model_id!r} requires a resolved profile for presentation."
        )
    spec = resolved_profile.spec
    _, variant = _depth_anything_v2_profile_values(resolved_profile)
    device = str(getattr(resolved_profile, "device", "cpu"))
    precision_value = getattr(getattr(resolved_profile, "precision", None), "value", "fp32")
    precision = str(precision_value).lower()
    if device.startswith("cpu"):
        precision = "fp32"
    elif precision not in {"fp32", "fp16"}:
        precision = "fp16"
    precision_label = "FP16" if precision == "fp16" else "FP32"
    tooltip = (
        f"Automatic Depth Anything V2 selection: {variant['model_label']}, "
        f"input size {variant['input_size']}, {precision_label}. Output is relative "
        "depth; 3D scale is arbitrary and measurements are unavailable."
    )
    warning = getattr(resolved_profile, "warning", None)
    if warning:
        tooltip += f"\nProfile: {warning}"
    return ModelProfileMetadata(
        variant_label=str(variant["model_label"]),
        detail_label=f"{int(variant['input_size'])} px · Relative depth",
        precision_label=precision_label,
        point_budget=max(1, int(spec.point_budget)),
        live_point_budget=max(1, int(spec.live_point_budget)),
        tooltip=tooltip,
    )


_VIDEO_DEPTH_ANYTHING_PROFILE_VARIANTS: dict[str, dict[str, object]] = {
    "efficient": {
        "encoder": "vits",
        "checkpoint": "checkpoints/video-depth-anything/metric_video_depth_anything_vits.pth",
        "model_label": "Metric Video Depth Anything Small",
        "input_size": 392,
        "max_resolution": 960,
    },
    "balanced": {
        "encoder": "vitb",
        "checkpoint": "checkpoints/video-depth-anything/metric_video_depth_anything_vitb.pth",
        "model_label": "Metric Video Depth Anything Base",
        "input_size": 448,
        "max_resolution": 1280,
    },
    "ultra": {
        "encoder": "vitl",
        "checkpoint": "checkpoints/video-depth-anything/metric_video_depth_anything_vitl.pth",
        "model_label": "Metric Video Depth Anything Large",
        "input_size": 518,
        "max_resolution": 1920,
    },
}


def _video_depth_anything_profile_values(resolved_profile: Any) -> tuple[str, dict[str, object]]:
    effective = str(getattr(getattr(resolved_profile, "effective", None), "value", "efficient"))
    if effective not in _VIDEO_DEPTH_ANYTHING_PROFILE_VARIANTS:
        effective = "efficient"
    return effective, dict(_VIDEO_DEPTH_ANYTHING_PROFILE_VARIANTS[effective])


def build_video_depth_anything_runtime_config(
    descriptor: ModelDescriptor,
    context: ModelRuntimeContext,
) -> dict[str, object]:
    """Resolve metric streaming VDA variants and conservative cache sizes."""

    resolved = context.resolved_profile
    if resolved is None or not hasattr(resolved, "spec"):
        raise ValueError(
            f"Model {descriptor.model_id!r} requires a resolved compute profile."
        )
    spec = resolved.spec
    effective, variant = _video_depth_anything_profile_values(resolved)
    device = str(getattr(resolved, "device", "cpu"))
    precision_value = getattr(getattr(resolved, "precision", None), "value", "fp32")
    precision = str(precision_value).lower()
    if device.startswith("cpu"):
        precision = "fp32"
    elif precision not in {"fp32", "fp16"}:
        precision = "fp16"
    config = dict(descriptor.default_config)
    config.update(variant)
    config.update(
        {
            "profile": effective,
            "device": device,
            "precision": precision,
            "point_budget": max(1, int(spec.point_budget)),
            "live_point_budget": max(1, int(spec.live_point_budget)),
            "warmup_size": max(64, min(256, int(spec.warmup_size))),
            "warmup_runs": 1,
            "allow_tf32": bool(spec.allow_tf32),
            "matmul_precision": str(spec.matmul_precision),
            "hardware_tier": effective,
        }
    )
    if context.point_budget is not None:
        config["point_budget"] = max(1, int(context.point_budget))
    config.update(dict(context.overrides))
    return validate_runtime_config(config)


def build_video_depth_anything_profile_metadata(
    descriptor: ModelDescriptor,
    resolved_profile: Any,
) -> ModelProfileMetadata:
    if resolved_profile is None or not hasattr(resolved_profile, "spec"):
        raise ValueError(
            f"Model {descriptor.model_id!r} requires a resolved profile for presentation."
        )
    spec = resolved_profile.spec
    _, variant = _video_depth_anything_profile_values(resolved_profile)
    device = str(getattr(resolved_profile, "device", "cpu"))
    precision_value = getattr(getattr(resolved_profile, "precision", None), "value", "fp32")
    precision = str(precision_value).lower()
    if device.startswith("cpu"):
        precision = "fp32"
    elif precision not in {"fp32", "fp16"}:
        precision = "fp16"
    precision_label = "FP16" if precision == "fp16" else "FP32"
    tooltip = (
        f"Automatic Video Depth Anything selection: {variant['model_label']}, "
        f"streaming input size {variant['input_size']}, {precision_label}. The "
        "upstream streaming path is experimental; temporal caches persist through "
        "Pause/Resume and reset on Stop or source changes."
    )
    warning = getattr(resolved_profile, "warning", None)
    if warning:
        tooltip += f"\nProfile: {warning}"
    return ModelProfileMetadata(
        variant_label=str(variant["model_label"]),
        detail_label=f"{int(variant['input_size'])} px · Temporal metric",
        precision_label=precision_label,
        point_budget=max(1, int(spec.point_budget)),
        live_point_budget=max(1, int(spec.live_point_budget)),
        tooltip=tooltip,
    )


_MARIGOLD_PROFILE_VARIANTS: dict[str, dict[str, object]] = {
    "efficient": {
        "model_label": "Marigold Depth v1.1 · Fast",
        "processing_resolution": 512,
        "num_inference_steps": 1,
        "ensemble_size": 1,
        "output_uncertainty": False,
        "model_cpu_offload": True,
        "attention_slicing": True,
        "vae_slicing": True,
        "vae_tiling": True,
        "inference_timeout_ms": 120_000,
        "stop_timeout_ms": 120_000,
    },
    "balanced": {
        "model_label": "Marigold Depth v1.1 · Balanced",
        "processing_resolution": 640,
        "num_inference_steps": 4,
        "ensemble_size": 1,
        "output_uncertainty": False,
        "model_cpu_offload": False,
        "attention_slicing": True,
        "vae_slicing": True,
        "vae_tiling": True,
        "inference_timeout_ms": 120_000,
        "stop_timeout_ms": 120_000,
    },
    "ultra": {
        "model_label": "Marigold Depth v1.1 · Precision",
        "processing_resolution": 768,
        "num_inference_steps": 10,
        "ensemble_size": 3,
        "output_uncertainty": True,
        "model_cpu_offload": False,
        "attention_slicing": False,
        "vae_slicing": True,
        "vae_tiling": True,
        "inference_timeout_ms": 180_000,
        "stop_timeout_ms": 180_000,
    },
}


def _marigold_profile_values(resolved_profile: Any) -> tuple[str, dict[str, object]]:
    effective = str(
        getattr(getattr(resolved_profile, "effective", None), "value", "efficient")
    )
    if effective not in _MARIGOLD_PROFILE_VARIANTS:
        effective = "efficient"
    return effective, dict(_MARIGOLD_PROFILE_VARIANTS[effective])


def build_marigold_runtime_config(
    descriptor: ModelDescriptor,
    context: ModelRuntimeContext,
) -> dict[str, object]:
    """Resolve conservative diffusion-depth presets through the registry."""

    resolved = context.resolved_profile
    if resolved is None or not hasattr(resolved, "spec"):
        raise ValueError(
            f"Model {descriptor.model_id!r} requires a resolved compute profile."
        )
    spec = resolved.spec
    effective, variant = _marigold_profile_values(resolved)
    device = str(getattr(resolved, "device", "cpu"))
    precision_value = getattr(getattr(resolved, "precision", None), "value", "fp32")
    precision = str(precision_value).lower()
    if device.startswith("cpu"):
        precision = "fp32"
        variant["model_cpu_offload"] = False
        variant["processing_resolution"] = min(
            384, int(variant["processing_resolution"])
        )
        variant["num_inference_steps"] = 1
        variant["ensemble_size"] = 1
        variant["output_uncertainty"] = False
        variant["inference_timeout_ms"] = 300_000
        variant["stop_timeout_ms"] = 300_000
    elif precision not in {"fp32", "fp16"}:
        precision = "fp16"

    config = dict(descriptor.default_config)
    config.update(variant)
    config.update(
        {
            "profile": effective,
            "device": device,
            "precision": precision,
            "point_budget": max(1, int(spec.point_budget)),
            "live_point_budget": max(1, min(int(spec.live_point_budget), 100_000)),
            "warmup_size": max(64, min(256, int(spec.warmup_size))),
            "warmup_runs": 1,
            "warmup_inference_steps": 1,
            "allow_tf32": bool(spec.allow_tf32),
            "matmul_precision": str(spec.matmul_precision),
            "hardware_tier": effective,
        }
    )
    if context.point_budget is not None:
        config["point_budget"] = max(1, int(context.point_budget))
    config.update(dict(context.overrides))
    return validate_runtime_config(config)


def build_marigold_profile_metadata(
    descriptor: ModelDescriptor,
    resolved_profile: Any,
) -> ModelProfileMetadata:
    """Expose quality/speed trade-offs without adding Marigold UI branches."""

    if resolved_profile is None or not hasattr(resolved_profile, "spec"):
        raise ValueError(
            f"Model {descriptor.model_id!r} requires a resolved profile for presentation."
        )
    spec = resolved_profile.spec
    _, variant = _marigold_profile_values(resolved_profile)
    device = str(getattr(resolved_profile, "device", "cpu"))
    precision_value = getattr(getattr(resolved_profile, "precision", None), "value", "fp32")
    precision = str(precision_value).lower()
    if device.startswith("cpu"):
        precision = "fp32"
    elif precision not in {"fp32", "fp16"}:
        precision = "fp16"
    precision_label = "FP16" if precision == "fp16" else "FP32"
    steps = int(variant["num_inference_steps"])
    ensemble = int(variant["ensemble_size"])
    uncertainty = " + uncertainty" if bool(variant["output_uncertainty"]) else ""
    offload = " · CPU offload" if bool(variant["model_cpu_offload"]) else ""
    tooltip = (
        f"Automatic Marigold preset: {variant['model_label']}, "
        f"{variant['processing_resolution']} px, {steps} diffusion step(s), "
        f"ensemble {ensemble}, {precision_label}{offload}. Output is affine-invariant "
        "relative depth and is intended for quality-oriented snapshot/offline use."
    )
    warning = getattr(resolved_profile, "warning", None)
    if warning:
        tooltip += f"\nProfile: {warning}"
    return ModelProfileMetadata(
        variant_label=str(variant["model_label"]),
        detail_label=(
            f"{int(variant['processing_resolution'])} px · {steps} step(s){uncertainty}"
        ),
        precision_label=precision_label,
        point_budget=max(1, int(spec.point_budget)),
        live_point_budget=max(1, min(int(spec.live_point_budget), 100_000)),
        tooltip=tooltip,
    )

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
