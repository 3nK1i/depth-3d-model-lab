"""Apple Depth Pro adapter for zero-shot metric monocular depth.

The official package and PyTorch are imported lazily inside :meth:`load`, so
selecting other models never imports Depth Pro or initializes CUDA. The adapter
runs only in the isolated inference worker and returns the common
:class:`~core.models.result.GeometryResult` contract.
"""

from __future__ import annotations

import gc
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from core.models.base import BaseGeometryModel
from core.models.catalog import DEPTH_PRO_DISPLAY_LABEL, DEPTH_PRO_MODEL_ID
from core.models.registry import get_model_descriptor
from core.models.result import GeometryResult
from core.runtime.messages import FrameRequest

_DEFAULT_CHECKPOINT = "checkpoints/depth_pro.pt"
_DEFAULT_INTERNAL_RESOLUTION = 1536
_DEFAULT_POINT_BUDGET = 250_000
_DEFAULT_WARMUP_SIZE = 256
_DEFAULT_WARMUP_RUNS = 1


class DepthProDependencyError(RuntimeError):
    """Raised when the optional official Depth Pro package is unavailable."""


class DepthProCheckpointError(RuntimeError):
    """Raised when the official local checkpoint cannot be resolved."""


class DepthProConfigurationError(RuntimeError):
    """Raised when a requested runtime configuration is unsupported."""


class DepthProOutOfMemoryError(RuntimeError):
    """Raised with a concise recovery message after a CUDA OOM."""


class DepthProGeometryModel(BaseGeometryModel):
    """Official Apple Depth Pro wrapped in the common registry adapter API."""

    def __init__(
        self,
        *,
        factory: Callable[..., tuple[Any, Any]] | None = None,
        default_config: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        # Dependency injection keeps tests independent from external downloads.
        self._factory = factory
        self._default_depth_pro_config = default_config
        self._torch = torch_module

        self._model: Any | None = None
        self._transform: Callable[[np.ndarray], Any] | None = None
        self._device: Any | None = None
        self._loaded = False

        self._precision = "fp32"
        self._checkpoint = _DEFAULT_CHECKPOINT
        self._resolved_checkpoint = ""
        self._model_name = DEPTH_PRO_DISPLAY_LABEL
        self._internal_resolution = _DEFAULT_INTERNAL_RESOLUTION
        self._point_budget = _DEFAULT_POINT_BUDGET
        self._warmup_size = _DEFAULT_WARMUP_SIZE
        self._warmup_runs = _DEFAULT_WARMUP_RUNS
        self._allow_tf32 = False
        self._tf32_enabled = False
        self._matmul_precision = "high"
        self._hardware_tier = "efficient"

    def load(self, device: str, precision: str, config: dict[str, Any]) -> None:
        torch = self._torch or self._import_torch()
        factory, default_config = self._resolve_official_api()
        self._torch = torch

        requested_precision = str(precision).lower()
        if requested_precision not in {"fp32", "fp16"}:
            raise DepthProConfigurationError(
                f"Unsupported Depth Pro precision: {requested_precision!r}."
            )

        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise DepthProConfigurationError(
                f"CUDA device {device!r} was requested, but PyTorch cannot access CUDA."
            )
        if target.type == "cpu" and requested_precision == "fp16":
            requested_precision = "fp32"

        self._device = target
        self._precision = requested_precision
        self._checkpoint = str(config.get("checkpoint", _DEFAULT_CHECKPOINT))
        checkpoint_path = self._resolve_checkpoint_path(self._checkpoint)
        if not checkpoint_path.is_file():
            raise DepthProCheckpointError(
                "Depth Pro checkpoint is missing at "
                f"{checkpoint_path}. Run: python scripts/download_depth_pro_checkpoint.py"
            )
        self._resolved_checkpoint = str(checkpoint_path)
        self._model_name = str(config.get("model_label", DEPTH_PRO_DISPLAY_LABEL))
        self._internal_resolution = max(
            1,
            int(config.get("internal_resolution", _DEFAULT_INTERNAL_RESOLUTION)),
        )
        self._point_budget = max(
            1,
            int(config.get("point_budget", _DEFAULT_POINT_BUDGET)),
        )
        self._warmup_size = max(
            64,
            int(config.get("warmup_size", _DEFAULT_WARMUP_SIZE)),
        )
        self._warmup_runs = min(
            4,
            max(1, int(config.get("warmup_runs", _DEFAULT_WARMUP_RUNS))),
        )
        self._allow_tf32 = bool(config.get("allow_tf32", False))
        self._matmul_precision = str(config.get("matmul_precision", "high")).lower()
        if self._matmul_precision not in {"highest", "high", "medium"}:
            raise DepthProConfigurationError(
                f"Unsupported float32 matmul precision: {self._matmul_precision!r}."
            )
        self._hardware_tier = str(
            config.get("hardware_tier", config.get("profile", "efficient"))
        )
        self._configure_torch_runtime()

        depth_pro_config = self._checkpoint_config(default_config, checkpoint_path)
        dtype = torch.float16 if requested_precision == "fp16" else torch.float32
        try:
            # Build and load weights on CPU first. The official factory moves the
            # model to its requested device before applying FP16, which creates a
            # large transient FP32 CUDA allocation. CPU-first loading avoids that
            # peak and is materially safer on the 6 GiB development GPU.
            factory_device = torch.device("cpu") if target.type == "cuda" else target
            model, transform = factory(
                config=depth_pro_config,
                device=factory_device,
                precision=dtype,
            )
            if target.type == "cuda":
                model = model.to(target)
            if not callable(transform):
                raise TypeError("Depth Pro factory returned a non-callable transform.")
            if hasattr(model, "eval"):
                model.eval()
        except Exception as exc:  # noqa: BLE001 - enrich model/checkpoint errors
            self._release_model()
            if self._is_cuda_oom(exc):
                raise self._oom_error(exc) from exc
            raise RuntimeError(
                f"Failed to load {self._model_name} from {checkpoint_path}: {exc}"
            ) from exc

        self._model = model
        self._transform = transform
        self._loaded = True

    def warmup(self) -> None:
        self._require_loaded()
        dummy = np.zeros(
            (self._warmup_size, self._warmup_size, 3),
            dtype=np.uint8,
        )
        try:
            for _ in range(self._warmup_runs):
                self._infer_model(dummy)
        except Exception as exc:  # noqa: BLE001
            if self._is_cuda_oom(exc):
                raise self._oom_error(exc) from exc
            raise RuntimeError(f"Depth Pro warm-up failed: {exc}") from exc

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
            postprocess_ms = (time.perf_counter() - post_started) * 1000.0
            worker_total_ms = (time.perf_counter() - worker_started) * 1000.0
            return replace(
                result,
                postprocess_ms=postprocess_ms,
                worker_total_ms=worker_total_ms,
            )
        except Exception as exc:  # noqa: BLE001
            if self._is_cuda_oom(exc):
                self._clear_cuda_cache()
                raise self._oom_error(exc) from exc
            raise

    def unload(self) -> dict[str, Any]:
        return self._release_model()

    def get_metadata(self) -> dict[str, Any]:
        descriptor = get_model_descriptor(DEPTH_PRO_MODEL_ID)
        return descriptor.metadata_with_runtime(
            display_name=self._model_name,
            family=DEPTH_PRO_DISPLAY_LABEL,
            checkpoint=self._resolved_checkpoint or self._checkpoint,
            device=str(self._device) if self._device is not None else "",
            precision=self.precision_label,
            internal_resolution=self._internal_resolution,
            point_budget=self._point_budget,
            warmup_size=self._warmup_size,
            warmup_runs=self._warmup_runs,
            hardware_tier=self._hardware_tier,
            allow_tf32=self._tf32_enabled,
            matmul_precision=self._matmul_precision,
            focal_length_source="model_predicted",
            load_strategy="cpu_weights_then_device",
        )

    @property
    def precision_label(self) -> str:
        return "FP16" if self._precision == "fp16" else "FP32"

    # -- Internal model execution -----------------------------------------

    def _infer_model(
        self,
        bgr_image: np.ndarray,
    ) -> tuple[Mapping[str, Any], float, float]:
        torch = self._torch
        assert torch is not None
        assert self._model is not None
        assert self._transform is not None
        assert self._device is not None

        rgb = self._validated_rgb(bgr_image)
        preprocess_started = time.perf_counter()
        image_tensor = self._transform(rgb)
        if hasattr(image_tensor, "to"):
            dtype = torch.float16 if self._precision == "fp16" else torch.float32
            image_tensor = image_tensor.to(device=self._device, dtype=dtype)
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
            raise RuntimeError("Depth Pro returned an unexpected non-mapping output.")
        return output, latency_ms, preprocess_ms

    def _run_model_inference(self, image_tensor: Any) -> Mapping[str, Any]:
        torch = self._torch
        assert torch is not None
        assert self._model is not None

        inference_mode = getattr(torch, "inference_mode", None)
        context = inference_mode() if callable(inference_mode) else nullcontext()
        with context:
            return self._model.infer(image_tensor, f_px=None)

    @staticmethod
    def _validated_rgb(bgr_image: np.ndarray) -> np.ndarray:
        if not isinstance(bgr_image, np.ndarray):
            raise TypeError("Depth Pro expects a NumPy image.")
        if bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
            raise ValueError("Depth Pro expects an H×W×3 BGR image.")
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
        depth_value = output.get("depth")
        if depth_value is None:
            raise RuntimeError("Depth Pro output did not contain a depth map.")
        depth = np.squeeze(self._to_numpy(depth_value, dtype=np.float32))
        if depth.ndim != 2:
            raise RuntimeError(
                f"Expected a 2D Depth Pro depth map, got shape {tuple(depth.shape)}."
            )
        if depth.shape != frame.image.shape[:2]:
            raise RuntimeError(
                "Depth Pro output resolution does not match the input frame: "
                f"expected {frame.image.shape[:2]}, got {depth.shape}."
            )
        depth = np.ascontiguousarray(depth, dtype=np.float32)
        valid_mask = np.ascontiguousarray(np.isfinite(depth) & (depth > 0.0))

        focal_length = self._scalar(output.get("focallength_px"))
        intrinsics = None
        if focal_length is not None and np.isfinite(focal_length) and focal_length > 0.0:
            height, width = depth.shape
            intrinsics = np.array(
                [
                    [focal_length, 0.0, (width - 1.0) * 0.5],
                    [0.0, focal_length, (height - 1.0) * 0.5],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

        (
            allocated_gib,
            reserved_gib,
            peak_reserved_gib,
            device_used_gib,
            total_gib,
        ) = self._vram_usage()
        return GeometryResult(
            session_id=frame.session_id,
            frame_id=frame.frame_id,
            timestamp=time.time(),
            rgb=frame.image,
            depth=depth,
            intrinsics=intrinsics,
            valid_mask=valid_mask,
            point_stride=1,
            latency_ms=latency_ms,
            preprocess_ms=preprocess_ms,
            queue_wait_ms=queue_wait_ms,
            is_metric=True,
            model_id=DEPTH_PRO_MODEL_ID,
            model_name=self._model_name,
            precision=self.precision_label,
            depth_scale="metric",
            intrinsics_source="model_predicted" if intrinsics is not None else "none",
            confidence_kind="none",
            vram_used_gib=reserved_gib,
            vram_allocated_gib=allocated_gib,
            vram_reserved_gib=reserved_gib,
            vram_peak_reserved_gib=peak_reserved_gib,
            vram_device_used_gib=device_used_gib,
            vram_total_gib=total_gib,
        )

    def _to_numpy(self, value: Any, *, dtype: np.dtype) -> np.ndarray:
        torch = self._torch
        assert torch is not None
        if isinstance(value, torch.Tensor):
            array = value.detach().contiguous().cpu().numpy()
        else:
            array = np.asarray(value)
        return np.ascontiguousarray(array.astype(dtype, copy=False))

    def _scalar(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            array = self._to_numpy(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError, RuntimeError):
            return None
        if array.size != 1:
            return None
        return float(array[0])

    # -- Dependency and resource helpers ----------------------------------

    def _import_torch(self):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - project runtime has torch
            raise DepthProDependencyError(
                "PyTorch is required for Depth Pro. Install a compatible CUDA build first."
            ) from exc
        return torch

    def _resolve_official_api(self) -> tuple[Callable[..., tuple[Any, Any]], Any]:
        if self._factory is not None and self._default_depth_pro_config is not None:
            return self._factory, self._default_depth_pro_config
        try:
            from depth_pro import create_model_and_transforms
            from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        except ImportError as exc:
            raise DepthProDependencyError(
                "The official Depth Pro package is not installed. Run: "
                "python -m pip install -r requirements/models-depth-pro.txt"
            ) from exc
        self._factory = create_model_and_transforms
        self._default_depth_pro_config = DEFAULT_MONODEPTH_CONFIG_DICT
        return create_model_and_transforms, DEFAULT_MONODEPTH_CONFIG_DICT

    @staticmethod
    def _checkpoint_config(default_config: Any, checkpoint_path: Path) -> Any:
        try:
            return replace(default_config, checkpoint_uri=str(checkpoint_path))
        except TypeError as exc:
            raise DepthProConfigurationError(
                "The installed Depth Pro package exposes an incompatible configuration API."
            ) from exc

    @staticmethod
    def _resolve_checkpoint_path(value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        project_root = Path(__file__).resolve().parents[3]
        return project_root / path

    def _require_loaded(self) -> None:
        if not self._loaded or self._model is None or self._transform is None:
            raise RuntimeError("DepthProGeometryModel is not loaded.")

    def _configure_torch_runtime(self) -> None:
        torch = self._torch
        if torch is None:
            return
        setter = getattr(torch, "set_float32_matmul_precision", None)
        if callable(setter):
            try:
                setter(self._matmul_precision)
            except Exception:  # noqa: BLE001 - optional optimization
                pass

        if self._device is None or self._device.type != "cuda":
            return
        capability_major = 0
        try:
            capability_major = int(torch.cuda.get_device_capability(self._device)[0])
        except Exception:  # noqa: BLE001
            pass
        tf32_supported = capability_major >= 8
        checker = getattr(torch.cuda, "is_tf32_supported", None)
        if callable(checker):
            try:
                tf32_supported = bool(checker())
            except Exception:  # noqa: BLE001
                pass
        enabled = bool(self._allow_tf32 and tf32_supported)
        self._tf32_enabled = enabled
        try:
            torch.backends.cuda.matmul.allow_tf32 = enabled
        except Exception:  # noqa: BLE001
            pass
        try:
            torch.backends.cudnn.allow_tf32 = enabled
        except Exception:  # noqa: BLE001
            pass

    def _release_model(self) -> dict[str, Any]:
        started = time.perf_counter()
        device = self._device
        before_allocated, before_reserved, _, before_device_used, total = self._vram_usage()

        torch = self._torch
        if (
            torch is not None
            and device is not None
            and getattr(device, "type", None) == "cuda"
            and hasattr(torch, "cuda")
        ):
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
            except Exception:  # noqa: BLE001 - cleanup is best effort
                pass

        self._loaded = False
        model = self._model
        transform = self._transform
        self._model = None
        self._transform = None
        if model is not None:
            del model
        if transform is not None:
            del transform
        gc.collect()
        self._clear_cuda_cache()
        if (
            torch is not None
            and device is not None
            and getattr(device, "type", None) == "cuda"
            and hasattr(torch, "cuda")
        ):
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(device)
            except Exception:  # noqa: BLE001
                pass

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
        except Exception:  # noqa: BLE001 - preserve original failures
            pass

    def _vram_usage(
        self,
    ) -> tuple[
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
    ]:
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

    def _oom_error(self, _exc: BaseException) -> DepthProOutOfMemoryError:
        return DepthProOutOfMemoryError(
            f"{self._model_name} ran out of GPU memory. Stop inference, close other "
            "GPU workloads, choose Efficient, then restart the worker."
        )
