"""UniDepthV2 adapter for metric depth, direct geometry, and uncertainty.

The official UniDepth package and PyTorch are imported lazily inside
:meth:`load`, so other registered models remain independent from this optional
runtime. The adapter executes only inside the isolated inference worker and
returns the common :class:`~core.models.result.GeometryResult` contract.
"""

from __future__ import annotations

import gc
import importlib
import math
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from core.geometry.sampling import stride_for_point_budget
from core.models.base import BaseGeometryModel
from core.models.catalog import UNIDEPTH_V2_DISPLAY_LABEL, UNIDEPTH_V2_MODEL_ID
from core.models.registry import get_model_descriptor
from core.models.result import GeometryResult
from core.runtime.messages import FrameRequest

_DEFAULT_CHECKPOINT_DIR = "checkpoints/unidepth-v2/vits14"
_DEFAULT_REPOSITORY = "lpiccinelli/unidepth-v2-vits14"
_DEFAULT_VARIANT = "vits14"
_DEFAULT_RESOLUTION_LEVEL = 2
_DEFAULT_POINT_BUDGET = 250_000
_DEFAULT_WARMUP_SIZE = 256
_DEFAULT_WARMUP_RUNS = 1
_VALID_INTERPOLATION_MODES = {"bilinear", "bicubic", "nearest"}


def _safe_unidepth_paddings(
    original_shape: tuple[int, int],
    aspect_ratio_range: tuple[float, float] | list[float],
) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """Return non-negative padding for the official V2 preprocessing path.

    The pinned upstream implementation can round an unchanged aspect ratio one
    pixel downward for specific dimensions, producing a negative padding. This
    isolated guard keeps the same intended behavior while using ``ceil`` and an
    equality tolerance.
    """

    height, width = (int(original_shape[0]), int(original_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("UniDepthV2 input dimensions must be positive.")
    minimum_ratio, maximum_ratio = (
        float(aspect_ratio_range[0]),
        float(aspect_ratio_range[1]),
    )
    if minimum_ratio <= 0.0 or maximum_ratio < minimum_ratio:
        raise ValueError("UniDepthV2 aspect-ratio bounds are invalid.")

    original_ratio = width / height
    target_ratio = min(maximum_ratio, max(minimum_ratio, original_ratio))
    if math.isclose(original_ratio, target_ratio, rel_tol=1e-9, abs_tol=1e-9):
        return (0, 0, 0, 0), (height, width)

    if original_ratio > target_ratio:
        padded_width = width
        padded_height = max(height, math.ceil(width / target_ratio))
        pad_top = (padded_height - height) // 2
        pad_bottom = padded_height - height - pad_top
        return (0, 0, pad_top, pad_bottom), (padded_height, padded_width)

    padded_height = height
    padded_width = max(width, math.ceil(height * target_ratio))
    pad_left = (padded_width - width) // 2
    pad_right = padded_width - width - pad_left
    return (pad_left, pad_right, 0, 0), (padded_height, padded_width)


class UniDepthV2DependencyError(RuntimeError):
    """Raised when the optional official UniDepthV2 runtime is unavailable."""


class UniDepthV2CheckpointError(RuntimeError):
    """Raised when a registry-selected local model snapshot is unavailable."""


class UniDepthV2ConfigurationError(RuntimeError):
    """Raised when a requested runtime configuration is unsupported."""


class UniDepthV2OutOfMemoryError(RuntimeError):
    """Raised with a concise recovery message after a CUDA OOM."""


class UniDepthV2GeometryModel(BaseGeometryModel):
    """Official UniDepthV2 wrapped in the common registry adapter API."""

    def __init__(
        self,
        *,
        model_loader: Callable[..., Any] | None = None,
        torch_module: Any | None = None,
    ) -> None:
        # Dependency injection keeps unit tests independent from downloads.
        self._model_loader = model_loader
        self._torch = torch_module

        self._model: Any | None = None
        self._device: Any | None = None
        self._loaded = False

        self._precision = "fp32"
        self._checkpoint_dir = _DEFAULT_CHECKPOINT_DIR
        self._resolved_checkpoint_dir = ""
        self._repository = _DEFAULT_REPOSITORY
        self._variant = _DEFAULT_VARIANT
        self._model_name = UNIDEPTH_V2_DISPLAY_LABEL
        self._resolution_level = _DEFAULT_RESOLUTION_LEVEL
        self._interpolation_mode = "bilinear"
        self._point_budget = _DEFAULT_POINT_BUDGET
        self._warmup_size = _DEFAULT_WARMUP_SIZE
        self._warmup_runs = _DEFAULT_WARMUP_RUNS
        self._allow_tf32 = False
        self._tf32_enabled = False
        self._matmul_precision = "high"
        self._hardware_tier = "efficient"

    def load(self, device: str, precision: str, config: dict[str, Any]) -> None:
        torch = self._torch or self._import_torch()
        self._torch = torch

        requested_precision = str(precision).lower()
        if requested_precision not in {"fp32", "fp16"}:
            raise UniDepthV2ConfigurationError(
                f"Unsupported UniDepthV2 precision: {requested_precision!r}."
            )

        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise UniDepthV2ConfigurationError(
                f"CUDA device {device!r} was requested, but PyTorch cannot access CUDA."
            )
        if target.type == "cpu" and requested_precision == "fp16":
            requested_precision = "fp32"

        self._device = target
        self._precision = requested_precision
        self._checkpoint_dir = str(
            config.get("checkpoint_dir", _DEFAULT_CHECKPOINT_DIR)
        )
        checkpoint_dir = self._resolve_checkpoint_dir(self._checkpoint_dir)
        self._validate_checkpoint_dir(checkpoint_dir)
        self._resolved_checkpoint_dir = str(checkpoint_dir)
        self._repository = str(config.get("repository", _DEFAULT_REPOSITORY))
        self._variant = str(config.get("variant", _DEFAULT_VARIANT))
        self._model_name = str(
            config.get("model_label", UNIDEPTH_V2_DISPLAY_LABEL)
        )
        self._resolution_level = int(
            config.get("resolution_level", _DEFAULT_RESOLUTION_LEVEL)
        )
        if not 0 <= self._resolution_level < 10:
            raise UniDepthV2ConfigurationError(
                "UniDepthV2 resolution_level must be in the range [0, 10)."
            )
        self._interpolation_mode = str(
            config.get("interpolation_mode", "bilinear")
        ).lower()
        if self._interpolation_mode not in _VALID_INTERPOLATION_MODES:
            raise UniDepthV2ConfigurationError(
                "Unsupported UniDepthV2 interpolation mode: "
                f"{self._interpolation_mode!r}."
            )
        self._point_budget = max(
            1, int(config.get("point_budget", _DEFAULT_POINT_BUDGET))
        )
        self._warmup_size = max(
            64, int(config.get("warmup_size", _DEFAULT_WARMUP_SIZE))
        )
        self._warmup_runs = min(
            4, max(1, int(config.get("warmup_runs", _DEFAULT_WARMUP_RUNS)))
        )
        self._allow_tf32 = bool(config.get("allow_tf32", False))
        self._matmul_precision = str(
            config.get("matmul_precision", "high")
        ).lower()
        if self._matmul_precision not in {"highest", "high", "medium"}:
            raise UniDepthV2ConfigurationError(
                f"Unsupported float32 matmul precision: {self._matmul_precision!r}."
            )
        self._hardware_tier = str(
            config.get("hardware_tier", config.get("profile", "efficient"))
        )
        self._configure_torch_runtime()

        loader = self._model_loader or self._resolve_official_loader()
        try:
            model = loader(str(checkpoint_dir), local_files_only=True)
            # The official V2 infer path reads both attributes directly.
            model.resolution_level = self._resolution_level
            model.interpolation_mode = self._interpolation_mode
            if requested_precision == "fp16" and hasattr(model, "half"):
                # Convert on CPU before moving to CUDA to avoid a transient full
                # FP32 allocation on the 6 GiB development GPU.
                model = model.half()
            if hasattr(model, "to"):
                model = model.to(target)
            if hasattr(model, "eval"):
                model.eval()
        except Exception as exc:  # noqa: BLE001 - enrich model/checkpoint errors
            self._release_model()
            if self._is_cuda_oom(exc):
                raise self._oom_error(exc) from exc
            raise RuntimeError(
                f"Failed to load {self._model_name} from {checkpoint_dir}: {exc}"
            ) from exc

        self._model = model
        self._loaded = True

    def warmup(self) -> None:
        self._require_loaded()
        dummy = np.zeros((self._warmup_size, self._warmup_size, 3), dtype=np.uint8)
        try:
            for _ in range(self._warmup_runs):
                self._infer_model(dummy)
        except Exception as exc:  # noqa: BLE001
            if self._is_cuda_oom(exc):
                raise self._oom_error(exc) from exc
            raise RuntimeError(f"UniDepthV2 warm-up failed: {exc}") from exc

    def infer(self, frame: FrameRequest) -> GeometryResult:
        self._require_loaded()
        worker_started = time.perf_counter()
        queue_wait_ms = self._queue_wait_ms(frame)
        self._reset_peak_memory_stats()
        try:
            output, latency_ms, preprocess_ms = self._infer_model(frame.image)
            post_started = time.perf_counter()
            result = self._to_geometry_result(
                frame,
                output,
                latency_ms,
                preprocess_ms=preprocess_ms,
                queue_wait_ms=queue_wait_ms,
            )
            return replace(
                result,
                postprocess_ms=(time.perf_counter() - post_started) * 1000.0,
                worker_total_ms=(time.perf_counter() - worker_started) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            if self._is_cuda_oom(exc):
                self._clear_cuda_cache()
                raise self._oom_error(exc) from exc
            raise

    def unload(self) -> dict[str, Any]:
        return self._release_model()

    def get_metadata(self) -> dict[str, Any]:
        descriptor = get_model_descriptor(UNIDEPTH_V2_MODEL_ID)
        return descriptor.metadata_with_runtime(
            display_name=self._model_name,
            family=UNIDEPTH_V2_DISPLAY_LABEL,
            checkpoint=self._resolved_checkpoint_dir or self._checkpoint_dir,
            repository=self._repository,
            variant=self._variant,
            device=str(self._device) if self._device is not None else "",
            precision=self.precision_label,
            resolution_level=self._resolution_level,
            interpolation_mode=self._interpolation_mode,
            point_budget=self._point_budget,
            warmup_size=self._warmup_size,
            warmup_runs=self._warmup_runs,
            hardware_tier=self._hardware_tier,
            allow_tf32=self._tf32_enabled,
            matmul_precision=self._matmul_precision,
            uncertainty_semantics="relative_scale_invariant_log_error",
            load_strategy="local_snapshot_cpu_then_device",
        )

    @property
    def precision_label(self) -> str:
        return "FP16" if self._precision == "fp16" else "FP32"

    def _infer_model(
        self, bgr_image: np.ndarray
    ) -> tuple[Mapping[str, Any], float, float]:
        torch = self._torch
        assert torch is not None
        assert self._model is not None
        assert self._device is not None

        preprocess_started = time.perf_counter()
        rgb = self._validated_rgb(bgr_image)
        image_tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).contiguous()
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0

        if self._device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output = self._run_model_inference(image_tensor)
            end_event.record()
            torch.cuda.synchronize(self._device)
            latency_ms = float(start_event.elapsed_time(end_event))
        else:
            started = time.perf_counter()
            output = self._run_model_inference(image_tensor)
            latency_ms = (time.perf_counter() - started) * 1000.0

        if not isinstance(output, Mapping):
            raise RuntimeError("UniDepthV2 returned an unexpected non-mapping output.")
        return output, latency_ms, preprocess_ms

    def _run_model_inference(self, image_tensor: Any) -> Mapping[str, Any]:
        torch = self._torch
        assert torch is not None
        assert self._model is not None
        inference_mode = getattr(torch, "inference_mode", None)
        context = inference_mode() if callable(inference_mode) else nullcontext()
        with context:
            return self._model.infer(image_tensor)

    @staticmethod
    def _validated_rgb(bgr_image: np.ndarray) -> np.ndarray:
        if not isinstance(bgr_image, np.ndarray):
            raise TypeError("UniDepthV2 expects a NumPy image.")
        if bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
            raise ValueError("UniDepthV2 expects an H×W×3 BGR image.")
        if bgr_image.dtype != np.uint8:
            bgr_image = np.clip(bgr_image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(bgr_image[:, :, ::-1])

    def _to_geometry_result(
        self,
        frame: FrameRequest,
        output: Mapping[str, Any],
        latency_ms: float,
        *,
        preprocess_ms: float | None,
        queue_wait_ms: float | None,
    ) -> GeometryResult:
        depth = self._spatial_scalar(output.get("depth"), "depth")
        expected_shape = frame.image.shape[:2]
        if depth.shape != expected_shape:
            raise RuntimeError(
                "UniDepthV2 output resolution does not match the input frame: "
                f"expected {expected_shape}, got {depth.shape}."
            )
        depth = np.ascontiguousarray(depth, dtype=np.float32)

        uncertainty = self._spatial_scalar(output.get("confidence"), "confidence")
        if uncertainty.shape != depth.shape:
            raise RuntimeError(
                "UniDepthV2 confidence output must match the depth resolution."
            )
        uncertainty = np.ascontiguousarray(uncertainty, dtype=np.float32)

        valid_mask = np.ascontiguousarray(
            np.isfinite(depth)
            & (depth > 0.0)
            & np.isfinite(uncertainty),
            dtype=np.bool_,
        )

        points_value = output.get("points")
        if points_value is None:
            raise RuntimeError("UniDepthV2 output did not contain direct 3D points.")
        points_full = self._spatial_vector(points_value, "points")
        if points_full.shape[:2] != depth.shape:
            raise RuntimeError(
                "UniDepthV2 points output must match the depth resolution."
            )
        requested_budget = frame.point_budget or self._point_budget
        stride = stride_for_point_budget(
            depth.shape[0], depth.shape[1], max(1, int(requested_budget))
        )
        points = np.ascontiguousarray(
            points_full[::stride, ::stride], dtype=np.float32
        )

        intrinsics = None
        if output.get("intrinsics") is not None:
            intrinsics = np.squeeze(
                self._to_numpy(output["intrinsics"], dtype=np.float32)
            )
            if intrinsics.shape != (3, 3):
                raise RuntimeError(
                    "Expected UniDepthV2 intrinsics with shape 3×3, got "
                    f"{tuple(intrinsics.shape)}."
                )
            intrinsics = np.ascontiguousarray(intrinsics, dtype=np.float32)

        allocated, reserved, peak_reserved, device_used, total = self._vram_usage()
        return GeometryResult(
            session_id=frame.session_id,
            frame_id=frame.frame_id,
            timestamp=time.time(),
            rgb=frame.image,
            depth=depth,
            points=points,
            intrinsics=intrinsics,
            confidence=uncertainty,
            valid_mask=valid_mask,
            point_stride=stride,
            latency_ms=latency_ms,
            preprocess_ms=preprocess_ms,
            queue_wait_ms=queue_wait_ms,
            is_metric=True,
            model_id=UNIDEPTH_V2_MODEL_ID,
            model_name=self._model_name,
            precision=self.precision_label,
            depth_scale="metric",
            intrinsics_source="model_predicted" if intrinsics is not None else "none",
            confidence_kind="uncertainty",
            vram_used_gib=reserved,
            vram_allocated_gib=allocated,
            vram_reserved_gib=reserved,
            vram_peak_reserved_gib=peak_reserved,
            vram_device_used_gib=device_used,
            vram_total_gib=total,
        )

    def _spatial_scalar(self, value: Any, name: str) -> np.ndarray:
        if value is None:
            raise RuntimeError(f"UniDepthV2 output did not contain {name!r}.")
        array = np.squeeze(self._to_numpy(value, dtype=np.float32))
        if array.ndim != 2:
            raise RuntimeError(
                f"Expected a 2D UniDepthV2 {name} map, got shape {tuple(array.shape)}."
            )
        return array

    def _spatial_vector(self, value: Any, name: str) -> np.ndarray:
        array = np.squeeze(self._to_numpy(value, dtype=np.float32))
        if array.ndim != 3:
            raise RuntimeError(
                f"Expected a 3D UniDepthV2 {name} map, got shape {tuple(array.shape)}."
            )
        if array.shape[0] == 3:
            array = np.moveaxis(array, 0, -1)
        if array.shape[-1] != 3:
            raise RuntimeError(
                f"Expected UniDepthV2 {name} channels to equal 3, got {array.shape}."
            )
        return np.ascontiguousarray(array, dtype=np.float32)

    def _to_numpy(self, value: Any, *, dtype: np.dtype) -> np.ndarray:
        torch = self._torch
        assert torch is not None
        if isinstance(value, torch.Tensor):
            array = value.detach().contiguous().cpu().numpy()
        else:
            array = np.asarray(value)
        return np.ascontiguousarray(array.astype(dtype, copy=False))

    def _import_torch(self):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise UniDepthV2DependencyError(
                "PyTorch is required for UniDepthV2. Install a compatible CUDA build first."
            ) from exc
        return torch

    def _resolve_official_loader(self) -> Callable[..., Any]:
        try:
            from unidepth.models import UniDepthV2

            module = importlib.import_module(
                "unidepth.models.unidepthv2.unidepthv2"
            )
            module.get_paddings = _safe_unidepth_paddings
        except (ImportError, AttributeError) as exc:
            raise UniDepthV2DependencyError(
                "The UniDepthV2 runtime is not installed. Run: "
                "python scripts/install_unidepth_v2.py"
            ) from exc
        return UniDepthV2.from_pretrained

    @staticmethod
    def _resolve_checkpoint_dir(value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[3] / path

    @staticmethod
    def _validate_checkpoint_dir(path: Path) -> None:
        weights = (path / "model.safetensors", path / "pytorch_model.bin")
        if not path.is_dir() or not (path / "config.json").is_file() or not any(
            item.is_file() for item in weights
        ):
            raise UniDepthV2CheckpointError(
                "UniDepthV2 model snapshot is missing or incomplete at "
                f"{path}. Run: python scripts/download_unidepth_v2_models.py --variant auto"
            )

    def _require_loaded(self) -> None:
        if not self._loaded or self._model is None:
            raise RuntimeError("UniDepthV2GeometryModel is not loaded.")

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
        capability_major = 0
        try:
            capability_major = int(torch.cuda.get_device_capability(self._device)[0])
        except Exception:  # noqa: BLE001
            pass
        enabled = bool(self._allow_tf32 and capability_major >= 8)
        self._tf32_enabled = enabled
        for backend in ("matmul", "cudnn"):
            try:
                target = (
                    torch.backends.cuda.matmul
                    if backend == "matmul"
                    else torch.backends.cudnn
                )
                target.allow_tf32 = enabled
            except Exception:  # noqa: BLE001
                pass

    def _release_model(self) -> dict[str, Any]:
        started = time.perf_counter()
        device = self._device
        before_allocated, before_reserved, _, before_device_used, total = self._vram_usage()
        torch = self._torch
        if torch is not None and device is not None and getattr(device, "type", None) == "cuda":
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
            except Exception:  # noqa: BLE001
                pass
        self._loaded = False
        model = self._model
        self._model = None
        if model is not None:
            del model
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

    def _vram_usage(self) -> tuple[float | None, float | None, float | None, float | None, float | None]:
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
        peak_reserved = float(torch.cuda.max_memory_reserved(self._device)) / 1024**3
        total = float(torch.cuda.get_device_properties(self._device).total_memory) / 1024**3
        device_used = None
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self._device)
            total = float(total_bytes) / 1024**3
            device_used = float(total_bytes - free_bytes) / 1024**3
        except (AttributeError, RuntimeError):
            pass
        return allocated, reserved, peak_reserved, device_used, total

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
        except (AttributeError, RuntimeError):
            pass

    @staticmethod
    def _queue_wait_ms(frame: FrameRequest) -> float | None:
        if frame.submitted_at is None:
            return None
        elapsed = (time.monotonic() - frame.submitted_at) * 1000.0
        if elapsed < 0.0 or elapsed > 60_000.0:
            return None
        return elapsed

    def _is_cuda_oom(self, exc: BaseException) -> bool:
        torch = self._torch
        oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", ())
        return bool(oom_type) and isinstance(exc, oom_type)

    def _oom_error(self, _exc: BaseException) -> UniDepthV2OutOfMemoryError:
        return UniDepthV2OutOfMemoryError(
            f"{self._model_name} ran out of GPU memory. Stop inference, close other "
            "GPU workloads, keep the automatic profile, then restart the worker."
        )
