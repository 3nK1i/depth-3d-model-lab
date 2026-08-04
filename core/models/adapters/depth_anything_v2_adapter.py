"""Depth Anything V2 adapter for robust relative monocular depth.

The official source and checkpoint are loaded lazily inside the isolated worker.
Depth Anything V2 predicts relative inverse depth/disparity. This adapter converts
that output into a monotonic relative-distance map before projection, so near
surfaces remain near in the 3D viewport. The scale is intentionally arbitrary and
is never presented as metric or calibrated.
"""

from __future__ import annotations

import gc
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from core.geometry.projection import (
    depth_to_points,
    intrinsics_from_virtual_fov,
    scale_intrinsics,
)
from core.geometry.sampling import stride_for_point_budget
from core.models.base import BaseGeometryModel
from core.models.catalog import (
    DEPTH_ANYTHING_V2_DISPLAY_LABEL,
    DEPTH_ANYTHING_V2_MODEL_ID,
)
from core.models.registry import get_model_descriptor
from core.models.result import GeometryResult
from core.runtime.messages import FrameRequest

_DEFAULT_SOURCE_DIR = "third_party/Depth-Anything-V2"
_DEFAULT_CHECKPOINT = "checkpoints/depth-anything-v2/depth_anything_v2_vits.pth"
_DEFAULT_ENCODER = "vits"
_DEFAULT_MODEL_NAME = "Depth Anything V2 Small"
_DEFAULT_INPUT_SIZE = 518
_DEFAULT_VIRTUAL_FOV = 60.0
_DEFAULT_POINT_BUDGET = 250_000
_DEFAULT_WARMUP_SIZE = 256
_DEFAULT_WARMUP_RUNS = 1
_VALID_ENCODERS = {"vits", "vitb", "vitl"}
_MODEL_CONFIGS: dict[str, dict[str, object]] = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


class DepthAnythingV2DependencyError(RuntimeError):
    """Raised when the optional official runtime is unavailable."""


class DepthAnythingV2CheckpointError(RuntimeError):
    """Raised when the source checkout or selected checkpoint is unavailable."""


class DepthAnythingV2ConfigurationError(RuntimeError):
    """Raised when the registry supplies an unsupported configuration."""


class DepthAnythingV2OutOfMemoryError(RuntimeError):
    """Raised with a concise recovery message after CUDA OOM."""


class DepthAnythingV2GeometryModel(BaseGeometryModel):
    """Official Depth Anything V2 wrapped in the common adapter contract."""

    def __init__(
        self,
        *,
        model_loader: Callable[[str, str], Any] | None = None,
        checkpoint_loader: Callable[[Path], Any] | None = None,
        torch_module: Any | None = None,
    ) -> None:
        self._model_loader = model_loader
        self._checkpoint_loader = checkpoint_loader
        self._torch = torch_module
        self._model: Any | None = None
        self._device: Any | None = None
        self._loaded = False
        self._precision = "fp32"
        self._source_dir = _DEFAULT_SOURCE_DIR
        self._resolved_source_dir = ""
        self._checkpoint = _DEFAULT_CHECKPOINT
        self._resolved_checkpoint = ""
        self._encoder = _DEFAULT_ENCODER
        self._model_name = _DEFAULT_MODEL_NAME
        self._input_size = _DEFAULT_INPUT_SIZE
        self._virtual_fov_degrees = _DEFAULT_VIRTUAL_FOV
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
            raise DepthAnythingV2ConfigurationError(
                f"Unsupported Depth Anything V2 precision: {requested_precision!r}."
            )
        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise DepthAnythingV2ConfigurationError(
                f"CUDA device {device!r} was requested, but PyTorch cannot access CUDA."
            )
        if target.type == "cpu" and requested_precision == "fp16":
            requested_precision = "fp32"

        self._device = target
        self._precision = requested_precision
        self._source_dir = str(config.get("source_dir", _DEFAULT_SOURCE_DIR))
        source_dir = self._resolve_project_path(self._source_dir)
        if not (source_dir / "depth_anything_v2" / "dpt.py").is_file():
            raise DepthAnythingV2CheckpointError(
                "Depth Anything V2 source checkout is missing or incomplete at "
                f"{source_dir}. Run: python scripts/install_depth_anything_models.py"
            )
        self._resolved_source_dir = str(source_dir)

        self._checkpoint = str(config.get("checkpoint", _DEFAULT_CHECKPOINT))
        checkpoint_path = self._resolve_project_path(self._checkpoint)
        if not checkpoint_path.is_file():
            raise DepthAnythingV2CheckpointError(
                "Depth Anything V2 checkpoint is missing at "
                f"{checkpoint_path}. Run: python scripts/download_depth_anything_models.py --family image --variant auto"
            )
        self._resolved_checkpoint = str(checkpoint_path)

        self._encoder = str(config.get("encoder", _DEFAULT_ENCODER)).lower()
        if self._encoder not in _VALID_ENCODERS:
            raise DepthAnythingV2ConfigurationError(
                f"Unsupported Depth Anything V2 encoder: {self._encoder!r}."
            )
        self._model_name = str(config.get("model_label", _DEFAULT_MODEL_NAME))
        self._input_size = max(196, int(config.get("input_size", _DEFAULT_INPUT_SIZE)))
        self._virtual_fov_degrees = float(
            config.get("virtual_fov_degrees", _DEFAULT_VIRTUAL_FOV)
        )
        if not 1.0 <= self._virtual_fov_degrees < 179.0:
            raise DepthAnythingV2ConfigurationError(
                "Depth Anything V2 virtual_fov_degrees must be in [1, 179)."
            )
        self._point_budget = max(1, int(config.get("point_budget", _DEFAULT_POINT_BUDGET)))
        self._warmup_size = max(64, int(config.get("warmup_size", _DEFAULT_WARMUP_SIZE)))
        self._warmup_runs = min(4, max(1, int(config.get("warmup_runs", _DEFAULT_WARMUP_RUNS))))
        self._allow_tf32 = bool(config.get("allow_tf32", False))
        self._matmul_precision = str(config.get("matmul_precision", "high")).lower()
        if self._matmul_precision not in {"highest", "high", "medium"}:
            raise DepthAnythingV2ConfigurationError(
                f"Unsupported float32 matmul precision: {self._matmul_precision!r}."
            )
        self._hardware_tier = str(config.get("hardware_tier", config.get("profile", "efficient")))
        self._configure_torch_runtime()

        loader = self._model_loader or self._load_official_model
        checkpoint_loader = self._checkpoint_loader or self._load_checkpoint
        try:
            model = loader(str(source_dir), self._encoder)
            state_dict = self._extract_state_dict(checkpoint_loader(checkpoint_path))
            if not hasattr(model, "load_state_dict"):
                raise TypeError("Depth Anything V2 loader returned an invalid model object.")
            model.load_state_dict(state_dict, strict=True)
            # Keep weights FP32 and use CUDA autocast. This avoids model-wide half
            # conversion incompatibilities while retaining the requested mixed precision.
            if hasattr(model, "float"):
                model = model.float()
            if hasattr(model, "to"):
                model = model.to(target)
            if hasattr(model, "eval"):
                model.eval()
        except Exception as exc:  # noqa: BLE001
            self._release_model()
            if self._is_cuda_oom(exc):
                raise self._oom_error() from exc
            raise RuntimeError(
                f"Failed to load {self._model_name} from {checkpoint_path}: {exc}"
            ) from exc
        self._model = model
        self._loaded = True

    def warmup(self) -> None:
        self._require_loaded()
        dummy = np.zeros((self._warmup_size, self._warmup_size, 3), dtype=np.uint8)
        try:
            for _ in range(self._warmup_runs):
                self._infer_disparity(dummy)
        except Exception as exc:  # noqa: BLE001
            if self._is_cuda_oom(exc):
                raise self._oom_error() from exc
            raise RuntimeError(f"Depth Anything V2 warm-up failed: {exc}") from exc

    def infer(self, frame: FrameRequest) -> GeometryResult:
        self._require_loaded()
        worker_started = time.perf_counter()
        queue_wait_ms = self._queue_wait_ms(frame)
        self._reset_peak_memory_stats()
        try:
            disparity, latency_ms, preprocess_ms = self._infer_disparity(frame.image)
            post_started = time.perf_counter()
            result = self._to_geometry_result(
                frame,
                disparity,
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
        descriptor = get_model_descriptor(DEPTH_ANYTHING_V2_MODEL_ID)
        return descriptor.metadata_with_runtime(
            display_name=self._model_name,
            family=DEPTH_ANYTHING_V2_DISPLAY_LABEL,
            source_dir=self._resolved_source_dir or self._source_dir,
            checkpoint=self._resolved_checkpoint or self._checkpoint,
            encoder=self._encoder,
            device=str(self._device) if self._device is not None else "",
            precision=self.precision_label,
            input_size=self._input_size,
            virtual_fov_degrees=self._virtual_fov_degrees,
            point_budget=self._point_budget,
            depth_representation="relative_distance_from_inverse_depth",
            intrinsics_source="estimated",
            scale_status="relative_not_metric",
            hardware_tier=self._hardware_tier,
            allow_tf32=self._tf32_enabled,
            matmul_precision=self._matmul_precision,
        )

    @property
    def precision_label(self) -> str:
        return "FP16" if self._precision == "fp16" else "FP32"

    def _infer_disparity(self, bgr_image: np.ndarray) -> tuple[np.ndarray, float, float]:
        torch = self._torch
        assert torch is not None and self._model is not None and self._device is not None
        image = self._validated_bgr(bgr_image)
        preprocess_started = time.perf_counter()
        # Official infer_image performs its own resize/normalization. Validation and
        # contiguity are the only adapter-side preprocessing steps.
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
        if self._device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output = self._run_inference(image)
            end_event.record()
            torch.cuda.synchronize(self._device)
            latency_ms = float(start_event.elapsed_time(end_event))
        else:
            started = time.perf_counter()
            output = self._run_inference(image)
            latency_ms = (time.perf_counter() - started) * 1000.0
        disparity = np.squeeze(self._to_numpy(output))
        if disparity.ndim != 2:
            raise RuntimeError(
                f"Expected a 2D Depth Anything V2 output, got {tuple(disparity.shape)}."
            )
        if disparity.shape != image.shape[:2]:
            disparity = cv2.resize(
                disparity,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return np.ascontiguousarray(disparity, dtype=np.float32), latency_ms, preprocess_ms

    def _run_inference(self, image: np.ndarray) -> Any:
        torch = self._torch
        assert torch is not None and self._model is not None
        inference_mode = getattr(torch, "inference_mode", None)
        inference_context = inference_mode() if callable(inference_mode) else nullcontext()
        with inference_context:
            with self._autocast_context():
                return self._model.infer_image(image, self._input_size)

    def _to_geometry_result(
        self,
        frame: FrameRequest,
        disparity: np.ndarray,
        latency_ms: float,
        *,
        preprocess_ms: float | None,
        queue_wait_ms: float | None,
        point_budget: int,
    ) -> GeometryResult:
        depth, valid_mask = self._relative_depth_from_disparity(disparity)
        height, width = depth.shape
        intrinsics = intrinsics_from_virtual_fov(width, height, self._virtual_fov_degrees)
        stride = stride_for_point_budget(height, width, point_budget)
        sampled_intrinsics = scale_intrinsics(intrinsics, 1.0 / float(stride))
        points = depth_to_points(depth[::stride, ::stride], intrinsics=sampled_intrinsics)
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
            valid_mask=valid_mask,
            point_stride=stride,
            latency_ms=latency_ms,
            preprocess_ms=preprocess_ms,
            queue_wait_ms=queue_wait_ms,
            is_metric=False,
            model_id=DEPTH_ANYTHING_V2_MODEL_ID,
            model_name=self._model_name,
            precision=self.precision_label,
            depth_scale="relative",
            intrinsics_source="estimated",
            confidence_kind="none",
            vram_used_gib=reserved,
            vram_allocated_gib=allocated,
            vram_reserved_gib=reserved,
            vram_peak_reserved_gib=peak,
            vram_device_used_gib=device_used,
            vram_total_gib=total,
        )

    @staticmethod
    def _relative_depth_from_disparity(disparity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        array = np.asarray(disparity, dtype=np.float32)
        valid = np.isfinite(array) & (array > 0.0)
        depth = np.zeros_like(array, dtype=np.float32)
        if not np.any(valid):
            return np.ascontiguousarray(depth), np.ascontiguousarray(valid)
        values = array[valid]
        low, high = np.percentile(values, [2.0, 98.0])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low + 1e-8:
            depth[valid] = 1.0
        else:
            normalized = np.clip((array - low) / (high - low), 0.0, 1.0)
            # Depth Anything predicts inverse depth: high values are near. Map to
            # stable relative distance in arbitrary units, preserving ordering.
            depth[valid] = 1.0 / (0.05 + normalized[valid])
        return np.ascontiguousarray(depth), np.ascontiguousarray(valid)

    @staticmethod
    def _validated_bgr(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError("Depth Anything V2 expects a NumPy image.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Depth Anything V2 expects an H×W×3 BGR image.")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image)

    def _load_official_model(self, source_dir: str, encoder: str) -> Any:
        source = str(Path(source_dir).resolve())
        # Keep the official checkout first during lazy import. This is especially
        # important for Video Depth Anything, whose upstream code imports a
        # top-level ``utils`` package from its own repository.
        if source in sys.path:
            sys.path.remove(source)
        sys.path.insert(0, source)
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except (ImportError, ModuleNotFoundError) as exc:
            raise DepthAnythingV2DependencyError(
                "Depth Anything V2 could not be imported. Run: "
                "python scripts/install_depth_anything_models.py"
            ) from exc
        return DepthAnythingV2(**dict(_MODEL_CONFIGS[encoder]))

    def _load_checkpoint(self, path: Path) -> Any:
        torch = self._torch
        assert torch is not None
        try:
            return torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(str(path), map_location="cpu")

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> Mapping[str, Any]:
        if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, Mapping):
            raise DepthAnythingV2CheckpointError(
                "Depth Anything V2 checkpoint does not contain a state dictionary."
            )
        return checkpoint

    def _autocast_context(self):
        torch = self._torch
        assert torch is not None
        if self._device is None or self._device.type != "cuda" or self._precision != "fp16":
            return nullcontext()
        autocast = getattr(torch, "autocast", None)
        if callable(autocast):
            return autocast(device_type="cuda", dtype=torch.float16)
        legacy = getattr(getattr(torch.cuda, "amp", None), "autocast", None)
        return legacy(dtype=torch.float16) if callable(legacy) else nullcontext()

    def _import_torch(self):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise DepthAnythingV2DependencyError(
                "PyTorch is required for Depth Anything V2."
            ) from exc
        return torch

    @staticmethod
    def _resolve_project_path(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else Path(__file__).resolve().parents[3] / path

    def _require_loaded(self) -> None:
        if not self._loaded or self._model is None:
            raise RuntimeError("DepthAnythingV2GeometryModel is not loaded.")

    def _to_numpy(self, value: Any) -> np.ndarray:
        torch = self._torch
        assert torch is not None
        if isinstance(value, torch.Tensor):
            value = value.detach().contiguous().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

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
        if torch is None or self._device is None or self._device.type != "cuda" or not torch.cuda.is_available():
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
        except (AttributeError, RuntimeError):
            pass
        return allocated, reserved, peak, used, total

    def _reset_peak_memory_stats(self) -> None:
        torch = self._torch
        if torch is None or self._device is None or self._device.type != "cuda" or not torch.cuda.is_available():
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
        return elapsed if 0.0 <= elapsed <= 60_000.0 else None

    def _is_cuda_oom(self, exc: BaseException) -> bool:
        torch = self._torch
        oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", ())
        return bool(oom_type) and isinstance(exc, oom_type)

    def _oom_error(self) -> DepthAnythingV2OutOfMemoryError:
        return DepthAnythingV2OutOfMemoryError(
            f"{self._model_name} ran out of GPU memory. Stop inference, close other "
            "GPU workloads, choose Efficient, then restart the worker."
        )
