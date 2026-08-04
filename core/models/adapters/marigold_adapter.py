"""Marigold diffusion-depth adapter.

The adapter uses the official Diffusers ``MarigoldDepthPipeline`` from a pinned
local Hugging Face snapshot. Marigold predicts affine-invariant relative depth,
so geometry scale is intentionally arbitrary and camera intrinsics are estimated
until calibration support is added in a later phase.
"""

from __future__ import annotations

import gc
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from core.geometry.projection import (
    depth_to_points,
    intrinsics_from_virtual_fov,
    scale_intrinsics,
)
from core.geometry.sampling import stride_for_point_budget
from core.models.base import BaseGeometryModel
from core.models.catalog import MARIGOLD_DISPLAY_LABEL, MARIGOLD_MODEL_ID
from core.models.registry import get_model_descriptor
from core.models.result import GeometryResult
from core.runtime.messages import FrameRequest

_DEFAULT_REPOSITORY = "prs-eth/marigold-depth-v1-1"
_DEFAULT_CHECKPOINT_DIR = "checkpoints/marigold/depth-v1-1"
_DEFAULT_MODEL_NAME = "Marigold Depth v1.1"
_DEFAULT_WEIGHT_VARIANT = "fp16"
_DEFAULT_PROCESSING_RESOLUTION = 512
_DEFAULT_INFERENCE_STEPS = 1
_DEFAULT_ENSEMBLE_SIZE = 1
_DEFAULT_VIRTUAL_FOV = 60.0
_DEFAULT_POINT_BUDGET = 100_000
_DEFAULT_WARMUP_SIZE = 192
_DEFAULT_INFERENCE_TIMEOUT_MS = 120_000


class MarigoldDependencyError(RuntimeError):
    """Raised when the optional Diffusers runtime is unavailable."""


class MarigoldCheckpointError(RuntimeError):
    """Raised when the pinned local model snapshot is missing or incomplete."""


class MarigoldConfigurationError(RuntimeError):
    """Raised when the registry supplies an unsupported configuration."""


class MarigoldOutOfMemoryError(RuntimeError):
    """Raised with a concise recovery message after CUDA OOM."""


class MarigoldGeometryModel(BaseGeometryModel):
    """Official Marigold depth pipeline wrapped in the common adapter contract."""

    def __init__(
        self,
        *,
        pipeline_loader: Callable[..., Any] | None = None,
        torch_module: Any | None = None,
    ) -> None:
        self._pipeline_loader = pipeline_loader
        self._torch = torch_module
        self._pipeline: Any | None = None
        self._device: Any | None = None
        self._loaded = False
        self._precision = "fp32"
        self._repository = _DEFAULT_REPOSITORY
        self._checkpoint_dir = _DEFAULT_CHECKPOINT_DIR
        self._resolved_checkpoint_dir = ""
        self._model_name = _DEFAULT_MODEL_NAME
        self._weight_variant = _DEFAULT_WEIGHT_VARIANT
        self._prediction_type = "depth"
        self._processing_resolution = _DEFAULT_PROCESSING_RESOLUTION
        self._num_inference_steps = _DEFAULT_INFERENCE_STEPS
        self._ensemble_size = _DEFAULT_ENSEMBLE_SIZE
        self._output_uncertainty = False
        self._model_cpu_offload = False
        self._offload_enabled = False
        self._attention_slicing = True
        self._vae_slicing = True
        self._vae_tiling = True
        self._seed = 0
        self._virtual_fov_degrees = _DEFAULT_VIRTUAL_FOV
        self._point_budget = _DEFAULT_POINT_BUDGET
        self._warmup_size = _DEFAULT_WARMUP_SIZE
        self._warmup_runs = 1
        self._warmup_inference_steps = 1
        self._allow_tf32 = False
        self._tf32_enabled = False
        self._matmul_precision = "high"
        self._hardware_tier = "efficient"
        self._inference_timeout_ms = _DEFAULT_INFERENCE_TIMEOUT_MS
        self._stop_timeout_ms = _DEFAULT_INFERENCE_TIMEOUT_MS

    def load(self, device: str, precision: str, config: dict[str, Any]) -> None:
        torch = self._torch or self._import_torch()
        self._torch = torch
        requested_precision = str(precision).lower()
        if requested_precision not in {"fp32", "fp16"}:
            raise MarigoldConfigurationError(
                f"Unsupported Marigold precision: {requested_precision!r}."
            )
        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise MarigoldConfigurationError(
                f"CUDA device {device!r} was requested, but PyTorch cannot access CUDA."
            )
        if target.type == "cpu" and requested_precision == "fp16":
            requested_precision = "fp32"

        self._device = target
        self._precision = requested_precision
        self._repository = str(config.get("repository", _DEFAULT_REPOSITORY))
        self._checkpoint_dir = str(
            config.get("checkpoint_dir", _DEFAULT_CHECKPOINT_DIR)
        )
        checkpoint_dir = self._resolve_project_path(self._checkpoint_dir)
        if not (checkpoint_dir / ".ready").is_file() or not (
            checkpoint_dir / "model_index.json"
        ).is_file():
            raise MarigoldCheckpointError(
                "Marigold snapshot is missing or incomplete at "
                f"{checkpoint_dir}. Run: python scripts/download_marigold_model.py"
            )
        self._resolved_checkpoint_dir = str(checkpoint_dir)
        self._model_name = str(config.get("model_label", _DEFAULT_MODEL_NAME))
        self._weight_variant = str(
            config.get("weight_variant", _DEFAULT_WEIGHT_VARIANT)
        ).strip()
        self._prediction_type = str(config.get("prediction_type", "depth")).lower()
        if self._prediction_type not in {"depth", "disparity"}:
            raise MarigoldConfigurationError(
                f"Unsupported Marigold prediction type: {self._prediction_type!r}."
            )
        self._processing_resolution = max(
            64,
            int(
                config.get(
                    "processing_resolution", _DEFAULT_PROCESSING_RESOLUTION
                )
            ),
        )
        self._num_inference_steps = min(
            50,
            max(1, int(config.get("num_inference_steps", _DEFAULT_INFERENCE_STEPS))),
        )
        self._ensemble_size = min(
            10,
            max(1, int(config.get("ensemble_size", _DEFAULT_ENSEMBLE_SIZE))),
        )
        if self._ensemble_size == 2:
            raise MarigoldConfigurationError(
                "Marigold ensemble_size must be 1 or at least 3."
            )
        self._output_uncertainty = bool(config.get("output_uncertainty", False))
        if self._output_uncertainty and self._ensemble_size < 3:
            self._output_uncertainty = False
        self._model_cpu_offload = bool(config.get("model_cpu_offload", False))
        if target.type != "cuda":
            self._model_cpu_offload = False
        self._attention_slicing = bool(config.get("attention_slicing", True))
        self._vae_slicing = bool(config.get("vae_slicing", True))
        self._vae_tiling = bool(config.get("vae_tiling", True))
        self._seed = int(config.get("seed", 0))
        self._virtual_fov_degrees = float(
            config.get("virtual_fov_degrees", _DEFAULT_VIRTUAL_FOV)
        )
        if not 1.0 <= self._virtual_fov_degrees < 179.0:
            raise MarigoldConfigurationError(
                "Marigold virtual_fov_degrees must be in [1, 179)."
            )
        self._point_budget = max(
            1, int(config.get("point_budget", _DEFAULT_POINT_BUDGET))
        )
        self._warmup_size = max(
            64, int(config.get("warmup_size", _DEFAULT_WARMUP_SIZE))
        )
        self._warmup_runs = min(2, max(1, int(config.get("warmup_runs", 1))))
        self._warmup_inference_steps = min(
            2, max(1, int(config.get("warmup_inference_steps", 1)))
        )
        self._allow_tf32 = bool(config.get("allow_tf32", False))
        self._matmul_precision = str(config.get("matmul_precision", "high")).lower()
        if self._matmul_precision not in {"highest", "high", "medium"}:
            raise MarigoldConfigurationError(
                f"Unsupported float32 matmul precision: {self._matmul_precision!r}."
            )
        self._hardware_tier = str(
            config.get("hardware_tier", config.get("profile", "efficient"))
        )
        self._inference_timeout_ms = min(
            600_000,
            max(
                5_000,
                int(
                    config.get(
                        "inference_timeout_ms", _DEFAULT_INFERENCE_TIMEOUT_MS
                    )
                ),
            ),
        )
        self._stop_timeout_ms = min(
            600_000,
            max(1_000, int(config.get("stop_timeout_ms", self._inference_timeout_ms))),
        )
        self._configure_torch_runtime()

        loader = self._pipeline_loader or self._load_official_pipeline
        dtype = torch.float16 if requested_precision == "fp16" else torch.float32
        try:
            pipeline = loader(
                str(checkpoint_dir),
                dtype=dtype,
                variant=(
                    self._weight_variant
                    if requested_precision == "fp16" and self._weight_variant
                    else None
                ),
            )
            self._configure_pipeline_memory(pipeline, target)
            pipeline_prediction_type = str(
                getattr(pipeline, "prediction_type", self._prediction_type)
            ).lower()
            if pipeline_prediction_type in {"depth", "disparity"}:
                self._prediction_type = pipeline_prediction_type
        except Exception as exc:  # noqa: BLE001
            self._release_model()
            if self._is_cuda_oom(exc):
                raise self._oom_error() from exc
            raise RuntimeError(
                f"Failed to load {self._model_name} from {checkpoint_dir}: {exc}"
            ) from exc
        self._pipeline = pipeline
        self._loaded = True

    def warmup(self) -> None:
        self._require_loaded()
        dummy = np.zeros((self._warmup_size, self._warmup_size, 3), dtype=np.uint8)
        try:
            for index in range(self._warmup_runs):
                self._infer_prediction(
                    dummy,
                    num_inference_steps=self._warmup_inference_steps,
                    ensemble_size=1,
                    output_uncertainty=False,
                    seed=self._seed + index,
                )
        except Exception as exc:  # noqa: BLE001
            if self._is_cuda_oom(exc):
                raise self._oom_error() from exc
            raise RuntimeError(f"Marigold warm-up failed: {exc}") from exc
        finally:
            self._clear_cuda_cache()

    def infer(self, frame: FrameRequest) -> GeometryResult:
        self._require_loaded()
        worker_started = time.perf_counter()
        queue_wait_ms = self._queue_wait_ms(frame)
        self._reset_peak_memory_stats()
        try:
            prediction, uncertainty, latency_ms, preprocess_ms = self._infer_prediction(
                frame.image,
                num_inference_steps=self._num_inference_steps,
                ensemble_size=self._ensemble_size,
                output_uncertainty=self._output_uncertainty,
                seed=self._seed,
            )
            post_started = time.perf_counter()
            result = self._to_geometry_result(
                frame,
                prediction,
                uncertainty,
                latency_ms,
                preprocess_ms=preprocess_ms,
                queue_wait_ms=queue_wait_ms,
                point_budget=max(1, int(frame.point_budget or self._point_budget)),
            )
            return replace(
                result,
                postprocess_ms=(time.perf_counter() - post_started) * 1000.0,
                worker_total_ms=(time.perf_counter() - worker_started) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            if self._is_cuda_oom(exc):
                self._clear_cuda_cache()
                raise self._oom_error() from exc
            raise

    def unload(self) -> dict[str, Any]:
        return self._release_model()

    def get_metadata(self) -> dict[str, Any]:
        descriptor = get_model_descriptor(MARIGOLD_MODEL_ID)
        return descriptor.metadata_with_runtime(
            display_name=self._model_name,
            family=MARIGOLD_DISPLAY_LABEL,
            repository=self._repository,
            checkpoint_dir=self._resolved_checkpoint_dir or self._checkpoint_dir,
            device=str(self._device) if self._device is not None else "",
            precision=self.precision_label,
            prediction_type=self._prediction_type,
            processing_resolution=self._processing_resolution,
            num_inference_steps=self._num_inference_steps,
            ensemble_size=self._ensemble_size,
            output_uncertainty=self._output_uncertainty,
            model_cpu_offload=self._offload_enabled,
            point_budget=self._point_budget,
            intrinsics_source="estimated",
            metric_scale_status="affine_invariant_relative",
            intended_usage="snapshot_offline_quality",
            supports_live=False,
            hardware_tier=self._hardware_tier,
            allow_tf32=self._tf32_enabled,
            matmul_precision=self._matmul_precision,
            inference_timeout_ms=self._inference_timeout_ms,
            stop_timeout_ms=self._stop_timeout_ms,
        )

    @property
    def precision_label(self) -> str:
        return "FP16" if self._precision == "fp16" else "FP32"

    def _infer_prediction(
        self,
        bgr_image: np.ndarray,
        *,
        num_inference_steps: int,
        ensemble_size: int,
        output_uncertainty: bool,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray | None, float, float]:
        torch = self._torch
        pipeline = self._pipeline
        assert torch is not None and pipeline is not None
        preprocess_started = time.perf_counter()
        bgr = self._validated_bgr(bgr_image)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        image = Image.fromarray(rgb)
        generator = self._make_generator(seed)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
        inference_mode = getattr(torch, "inference_mode", None)
        context = inference_mode() if callable(inference_mode) else nullcontext()
        started = time.perf_counter()
        with context:
            output = pipeline(
                image=image,
                num_inference_steps=num_inference_steps,
                ensemble_size=ensemble_size,
                processing_resolution=self._processing_resolution,
                match_input_resolution=True,
                batch_size=1,
                generator=generator,
                output_type="np",
                output_uncertainty=output_uncertainty,
                return_dict=True,
            )
        self._synchronize_cuda()
        latency_ms = (time.perf_counter() - started) * 1000.0
        prediction = self._extract_map(getattr(output, "prediction", None), "prediction")
        uncertainty_value = getattr(output, "uncertainty", None)
        uncertainty = (
            self._extract_map(uncertainty_value, "uncertainty")
            if uncertainty_value is not None
            else None
        )
        if prediction.shape != bgr.shape[:2]:
            raise RuntimeError(
                "Marigold output resolution does not match the input after "
                f"match_input_resolution=True: {prediction.shape} vs {bgr.shape[:2]}."
            )
        if uncertainty is not None and uncertainty.shape != prediction.shape:
            raise RuntimeError(
                "Marigold uncertainty shape does not match depth: "
                f"{uncertainty.shape} vs {prediction.shape}."
            )
        return prediction, uncertainty, latency_ms, preprocess_ms

    def _to_geometry_result(
        self,
        frame: FrameRequest,
        prediction: np.ndarray,
        uncertainty: np.ndarray | None,
        latency_ms: float,
        *,
        preprocess_ms: float | None,
        queue_wait_ms: float | None,
        point_budget: int,
    ) -> GeometryResult:
        depth, valid_mask = self._relative_depth(prediction)
        if uncertainty is not None:
            uncertainty = np.asarray(uncertainty, dtype=np.float32)
            uncertainty = np.where(np.isfinite(uncertainty), uncertainty, 1.0)
            uncertainty = np.clip(uncertainty, 0.0, 1.0)
            uncertainty = np.ascontiguousarray(uncertainty, dtype=np.float32)
        height, width = depth.shape
        intrinsics = intrinsics_from_virtual_fov(
            width, height, self._virtual_fov_degrees
        )
        stride = stride_for_point_budget(height, width, point_budget)
        points = depth_to_points(
            depth[::stride, ::stride],
            intrinsics=scale_intrinsics(intrinsics, 1.0 / float(stride)),
        )
        rgb = np.ascontiguousarray(frame.image[:, :, ::-1])
        colors = np.ascontiguousarray(rgb[::stride, ::stride], dtype=np.uint8)
        allocated, reserved, peak, device_used, total = self._vram_usage()
        return GeometryResult(
            session_id=frame.session_id,
            frame_id=frame.frame_id,
            timestamp=time.time(),
            rgb=frame.image,
            depth=depth,
            points=np.ascontiguousarray(points, dtype=np.float32),
            colors=colors,
            intrinsics=np.ascontiguousarray(intrinsics, dtype=np.float32),
            confidence=uncertainty,
            valid_mask=valid_mask,
            point_stride=stride,
            latency_ms=latency_ms,
            preprocess_ms=preprocess_ms,
            queue_wait_ms=queue_wait_ms,
            is_metric=False,
            model_id=MARIGOLD_MODEL_ID,
            model_name=self._model_name,
            precision=self.precision_label,
            depth_scale="relative",
            intrinsics_source="estimated",
            confidence_kind="uncertainty" if uncertainty is not None else "none",
            vram_used_gib=reserved,
            vram_allocated_gib=allocated,
            vram_reserved_gib=reserved,
            vram_peak_reserved_gib=peak,
            vram_device_used_gib=device_used,
            vram_total_gib=total,
        )

    def _relative_depth(self, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(prediction, dtype=np.float32)
        valid = np.isfinite(values)
        depth = np.zeros_like(values, dtype=np.float32)
        if not np.any(valid):
            return np.ascontiguousarray(depth), np.ascontiguousarray(valid)
        clipped = np.clip(values, 0.0, 1.0)
        if self._prediction_type == "disparity":
            # Larger disparity means nearer. Convert to monotonic relative distance.
            depth[valid] = 1.0 / (0.05 + clipped[valid])
        else:
            # Official Marigold depth predictions are affine-invariant in [0, 1].
            # Keep ordering and avoid exact zero during 3D projection.
            depth[valid] = 0.05 + 0.95 * clipped[valid]
        return np.ascontiguousarray(depth), np.ascontiguousarray(valid)

    def _configure_pipeline_memory(self, pipeline: Any, target: Any) -> None:
        if self._attention_slicing:
            method = getattr(pipeline, "enable_attention_slicing", None)
            if callable(method):
                method("max")
        vae = getattr(pipeline, "vae", None)
        if vae is not None and self._vae_slicing:
            method = getattr(vae, "enable_slicing", None)
            if callable(method):
                method()
        if vae is not None and self._vae_tiling:
            method = getattr(vae, "enable_tiling", None)
            if callable(method):
                method()
        if target.type == "cuda" and self._model_cpu_offload:
            method = getattr(pipeline, "enable_model_cpu_offload", None)
            if not callable(method):
                raise MarigoldDependencyError(
                    "Marigold CPU offload requires a compatible Diffusers and "
                    "Accelerate installation."
                )
            gpu_id = int(target.index or 0)
            try:
                method(gpu_id=gpu_id)
            except TypeError:
                method()
            self._offload_enabled = True
            return
        to_method = getattr(pipeline, "to", None)
        if not callable(to_method):
            raise TypeError("Marigold loader returned an invalid pipeline object.")
        to_method(target)
        self._offload_enabled = False

    def _load_official_pipeline(
        self,
        checkpoint_dir: str,
        *,
        dtype: Any,
        variant: str | None,
    ) -> Any:
        try:
            from diffusers import MarigoldDepthPipeline
        except (ImportError, ModuleNotFoundError) as exc:
            raise MarigoldDependencyError(
                "Marigold Diffusers runtime is unavailable. Run: "
                "python scripts/install_marigold.py"
            ) from exc
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }
        if variant:
            kwargs["variant"] = variant
        try:
            return MarigoldDepthPipeline.from_pretrained(checkpoint_dir, **kwargs)
        except TypeError as exc:
            # Newer Diffusers releases accept ``dtype`` as the public spelling;
            # retain compatibility with releases that still use ``torch_dtype``.
            if "torch_dtype" not in str(exc):
                raise
            kwargs["dtype"] = kwargs.pop("torch_dtype")
            return MarigoldDepthPipeline.from_pretrained(checkpoint_dir, **kwargs)

    def _make_generator(self, seed: int) -> Any | None:
        torch = self._torch
        if torch is None:
            return None
        generator_class = getattr(torch, "Generator", None)
        if generator_class is None:
            return None
        try:
            generator = generator_class(device="cpu")
        except TypeError:
            generator = generator_class()
        manual_seed = getattr(generator, "manual_seed", None)
        return manual_seed(int(seed)) if callable(manual_seed) else generator

    @staticmethod
    def _extract_map(value: Any, name: str) -> np.ndarray:
        if value is None:
            raise RuntimeError(f"Marigold output is missing {name}.")
        array = np.asarray(value, dtype=np.float32)
        array = np.squeeze(array)
        if array.ndim != 2:
            raise RuntimeError(
                f"Expected a 2D Marigold {name} map, got {tuple(array.shape)}."
            )
        return np.ascontiguousarray(array, dtype=np.float32)

    @staticmethod
    def _validated_bgr(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError("Marigold expects a NumPy image.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Marigold expects an H×W×3 BGR image.")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image)

    def _import_torch(self) -> Any:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise MarigoldDependencyError("PyTorch is required for Marigold.") from exc
        return torch

    @staticmethod
    def _resolve_project_path(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else Path(__file__).resolve().parents[3] / path

    def _require_loaded(self) -> None:
        if not self._loaded or self._pipeline is None:
            raise RuntimeError("MarigoldGeometryModel is not loaded.")

    def _configure_torch_runtime(self) -> None:
        torch = self._torch
        if torch is None:
            return
        setter = getattr(torch, "set_float32_matmul_precision", None)
        if callable(setter):
            try:
                setter(self._matmul_precision)
            except Exception:  # noqa: BLE001
                pass
        if self._device is None or self._device.type != "cuda":
            return
        supported = False
        try:
            supported = int(torch.cuda.get_device_capability(self._device)[0]) >= 8
        except Exception:  # noqa: BLE001
            pass
        self._tf32_enabled = bool(self._allow_tf32 and supported)
        try:
            torch.backends.cuda.matmul.allow_tf32 = self._tf32_enabled
            torch.backends.cudnn.allow_tf32 = self._tf32_enabled
        except Exception:  # noqa: BLE001
            pass

    def _release_model(self) -> dict[str, Any]:
        started = time.perf_counter()
        device = self._device
        before_allocated, before_reserved, _, before_device_used, total = self._vram_usage()
        self._loaded = False
        pipeline = self._pipeline
        self._pipeline = None
        self._offload_enabled = False
        if pipeline is not None:
            try:
                remove_hooks = getattr(pipeline, "remove_all_hooks", None)
                if callable(remove_hooks):
                    remove_hooks()
            except Exception:  # noqa: BLE001
                pass
            del pipeline
        gc.collect()
        self._clear_cuda_cache()
        after_allocated, after_reserved, _, after_device_used, _ = self._vram_usage()
        self._device = None
        return {
            "released": True,
            "device": str(device) if device is not None else "",
            "before_allocated_gib": before_allocated,
            "before_reserved_gib": before_reserved,
            "before_device_used_gib": before_device_used,
            "after_allocated_gib": after_allocated,
            "after_reserved_gib": after_reserved,
            "after_device_used_gib": after_device_used,
            "total_gib": total,
            "cleanup_time_ms": (time.perf_counter() - started) * 1000.0,
        }

    def _clear_cuda_cache(self) -> None:
        torch = self._torch
        if torch is None or not hasattr(torch, "cuda"):
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001
            pass

    def _synchronize_cuda(self) -> None:
        torch = self._torch
        if (
            torch is None
            or self._device is None
            or self._device.type != "cuda"
            or not torch.cuda.is_available()
        ):
            return
        try:
            torch.cuda.synchronize(self._device)
        except Exception:  # noqa: BLE001
            pass

    def _vram_usage(
        self,
    ) -> tuple[float | None, float | None, float | None, float | None, float | None]:
        torch = self._torch
        if (
            torch is None
            or self._device is None
            or self._device.type != "cuda"
            or not torch.cuda.is_available()
        ):
            return None, None, None, None, None
        allocated = float(torch.cuda.memory_allocated(self._device)) / 1024**3
        reserved = float(torch.cuda.memory_reserved(self._device)) / 1024**3
        peak = float(torch.cuda.max_memory_reserved(self._device)) / 1024**3
        total = float(torch.cuda.get_device_properties(self._device).total_memory) / 1024**3
        used = None
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self._device)
            total = float(total_bytes) / 1024**3
            used = float(total_bytes - free_bytes) / 1024**3
        except Exception:  # noqa: BLE001
            pass
        return allocated, reserved, peak, used, total

    def _reset_peak_memory_stats(self) -> None:
        torch = self._torch
        if (
            torch is None
            or self._device is None
            or self._device.type != "cuda"
            or not torch.cuda.is_available()
        ):
            return
        try:
            torch.cuda.reset_peak_memory_stats(self._device)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _queue_wait_ms(frame: FrameRequest) -> float | None:
        if frame.submitted_at is None:
            return None
        return max(0.0, (time.monotonic() - float(frame.submitted_at)) * 1000.0)

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        message = str(exc).lower()
        return "out of memory" in message and ("cuda" in message or "gpu" in message)

    def _oom_error(self) -> MarigoldOutOfMemoryError:
        return MarigoldOutOfMemoryError(
            "Marigold exhausted GPU memory. The model was released. Retry with "
            "the Efficient profile, close other GPU applications, or use an RTX "
            "5090 for the higher-quality presets."
        )
