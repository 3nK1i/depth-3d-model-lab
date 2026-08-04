"""Streaming Video Depth Anything adapter with persistent temporal caches.

The official experimental streaming API accepts one RGB frame at a time and
reuses temporal-attention hidden states. The adapter resets those caches on
source-session changes, resolution changes, frame-counter restarts, STOP, model
switches, worker recovery, and unload. Metric checkpoints are used so the
normalized result contract remains metric, while intrinsics stay explicitly
estimated until camera calibration is introduced.
"""

from __future__ import annotations

import gc
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from core.geometry.projection import depth_to_points, intrinsics_from_virtual_fov, scale_intrinsics
from core.geometry.sampling import stride_for_point_budget
from core.models.base import BaseGeometryModel
from core.models.catalog import VIDEO_DEPTH_ANYTHING_DISPLAY_LABEL, VIDEO_DEPTH_ANYTHING_MODEL_ID
from core.models.registry import get_model_descriptor
from core.models.result import GeometryResult
from core.runtime.messages import FrameRequest

_DEFAULT_SOURCE_DIR = "third_party/Video-Depth-Anything"
_DEFAULT_CHECKPOINT = "checkpoints/video-depth-anything/metric_video_depth_anything_vits.pth"
_DEFAULT_ENCODER = "vits"
_DEFAULT_MODEL_NAME = "Metric Video Depth Anything Small"
_DEFAULT_INPUT_SIZE = 392
_DEFAULT_MAX_RESOLUTION = 960
_DEFAULT_VIRTUAL_FOV = 60.0
_DEFAULT_MAX_DEPTH = 300.0
_DEFAULT_POINT_BUDGET = 100_000
_DEFAULT_WARMUP_SIZE = 224
_VALID_ENCODERS = {"vits", "vitb", "vitl"}
_MODEL_CONFIGS: dict[str, dict[str, object]] = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


class VideoDepthAnythingDependencyError(RuntimeError):
    pass


class VideoDepthAnythingCheckpointError(RuntimeError):
    pass


class VideoDepthAnythingConfigurationError(RuntimeError):
    pass


class VideoDepthAnythingOutOfMemoryError(RuntimeError):
    pass


class VideoDepthAnythingGeometryModel(BaseGeometryModel):
    """Official metric streaming VDA model behind the common adapter contract."""

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
        self._max_resolution = _DEFAULT_MAX_RESOLUTION
        self._virtual_fov_degrees = _DEFAULT_VIRTUAL_FOV
        self._max_depth_m = _DEFAULT_MAX_DEPTH
        self._point_budget = _DEFAULT_POINT_BUDGET
        self._warmup_size = _DEFAULT_WARMUP_SIZE
        self._allow_tf32 = False
        self._tf32_enabled = False
        self._matmul_precision = "high"
        self._hardware_tier = "efficient"
        self._last_session_id: int | None = None
        self._last_frame_id: int | None = None
        self._sequence_shape: tuple[int, int] | None = None
        self._sequence_resets = 0

    def load(self, device: str, precision: str, config: dict[str, Any]) -> None:
        torch = self._torch or self._import_torch()
        self._torch = torch
        requested_precision = str(precision).lower()
        if requested_precision not in {"fp32", "fp16"}:
            raise VideoDepthAnythingConfigurationError(
                f"Unsupported Video Depth Anything precision: {requested_precision!r}."
            )
        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise VideoDepthAnythingConfigurationError(
                f"CUDA device {device!r} was requested, but PyTorch cannot access CUDA."
            )
        if target.type == "cpu" and requested_precision == "fp16":
            requested_precision = "fp32"
        self._device = target
        self._precision = requested_precision

        self._source_dir = str(config.get("source_dir", _DEFAULT_SOURCE_DIR))
        source_dir = self._resolve_project_path(self._source_dir)
        if not (source_dir / "video_depth_anything" / "video_depth_stream.py").is_file():
            raise VideoDepthAnythingCheckpointError(
                "Video Depth Anything source checkout is missing or incomplete at "
                f"{source_dir}. Run: python scripts/install_depth_anything_models.py"
            )
        self._resolved_source_dir = str(source_dir)
        self._checkpoint = str(config.get("checkpoint", _DEFAULT_CHECKPOINT))
        checkpoint_path = self._resolve_project_path(self._checkpoint)
        if not checkpoint_path.is_file():
            raise VideoDepthAnythingCheckpointError(
                "Video Depth Anything checkpoint is missing at "
                f"{checkpoint_path}. Run: python scripts/download_depth_anything_models.py --family video --variant auto"
            )
        self._resolved_checkpoint = str(checkpoint_path)
        self._encoder = str(config.get("encoder", _DEFAULT_ENCODER)).lower()
        if self._encoder not in _VALID_ENCODERS:
            raise VideoDepthAnythingConfigurationError(
                f"Unsupported Video Depth Anything encoder: {self._encoder!r}."
            )
        self._model_name = str(config.get("model_label", _DEFAULT_MODEL_NAME))
        self._input_size = max(196, int(config.get("input_size", _DEFAULT_INPUT_SIZE)))
        self._max_resolution = max(0, int(config.get("max_resolution", _DEFAULT_MAX_RESOLUTION)))
        self._virtual_fov_degrees = float(config.get("virtual_fov_degrees", _DEFAULT_VIRTUAL_FOV))
        if not 1.0 <= self._virtual_fov_degrees < 179.0:
            raise VideoDepthAnythingConfigurationError(
                "Video Depth Anything virtual_fov_degrees must be in [1, 179)."
            )
        self._max_depth_m = float(config.get("max_depth_m", _DEFAULT_MAX_DEPTH))
        if self._max_depth_m <= 0.0:
            raise VideoDepthAnythingConfigurationError("max_depth_m must be positive.")
        self._point_budget = max(1, int(config.get("point_budget", _DEFAULT_POINT_BUDGET)))
        self._warmup_size = max(64, int(config.get("warmup_size", _DEFAULT_WARMUP_SIZE)))
        self._allow_tf32 = bool(config.get("allow_tf32", False))
        self._matmul_precision = str(config.get("matmul_precision", "high")).lower()
        if self._matmul_precision not in {"highest", "high", "medium"}:
            raise VideoDepthAnythingConfigurationError(
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
                raise TypeError("Video Depth Anything loader returned an invalid model object.")
            model.load_state_dict(state_dict, strict=True)
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
        self._reset_sequence_state(clear_cuda=False)

    def warmup(self) -> None:
        self._require_loaded()
        dummy = np.zeros((self._warmup_size, self._warmup_size, 3), dtype=np.uint8)
        try:
            self._infer_depth(dummy)
        except Exception as exc:  # noqa: BLE001
            if self._is_cuda_oom(exc):
                raise self._oom_error() from exc
            raise RuntimeError(f"Video Depth Anything warm-up failed: {exc}") from exc
        finally:
            # Warm-up must never become the first frame of the user's sequence.
            self._reset_sequence_state(clear_cuda=True)

    def infer(self, frame: FrameRequest) -> GeometryResult:
        self._require_loaded()
        worker_started = time.perf_counter()
        queue_wait_ms = self._queue_wait_ms(frame)
        self._reset_peak_memory_stats()
        self._prepare_sequence(frame)
        try:
            depth, latency_ms, preprocess_ms = self._infer_depth(frame.image)
            self._last_session_id = frame.session_id
            self._last_frame_id = frame.frame_id
            self._sequence_shape = frame.image.shape[:2]
            post_started = time.perf_counter()
            result = self._to_geometry_result(
                frame,
                depth,
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
                self._reset_sequence_state(clear_cuda=True)
                raise self._oom_error() from exc
            raise

    def reset_sequence(self) -> dict[str, Any]:
        before = self._vram_usage()
        reset = self._reset_sequence_state(clear_cuda=True)
        after = self._vram_usage()
        return {
            "sequence_reset": reset,
            "sequence_resets": self._sequence_resets,
            "before_reserved_gib": before[1],
            "after_reserved_gib": after[1],
            "after_device_used_gib": after[3],
            "total_gib": after[4],
        }

    def unload(self) -> dict[str, Any]:
        return self._release_model()

    def get_metadata(self) -> dict[str, Any]:
        descriptor = get_model_descriptor(VIDEO_DEPTH_ANYTHING_MODEL_ID)
        return descriptor.metadata_with_runtime(
            display_name=self._model_name,
            family=VIDEO_DEPTH_ANYTHING_DISPLAY_LABEL,
            source_dir=self._resolved_source_dir or self._source_dir,
            checkpoint=self._resolved_checkpoint or self._checkpoint,
            encoder=self._encoder,
            device=str(self._device) if self._device is not None else "",
            precision=self.precision_label,
            input_size=self._input_size,
            max_resolution=self._max_resolution,
            point_budget=self._point_budget,
            intrinsics_source="estimated",
            metric_scale_status="model_metric_estimated_camera_intrinsics",
            temporal_mode="official_streaming_cache",
            upstream_streaming_status="experimental",
            stateful_sequence=True,
            sequence_resets=self._sequence_resets,
            hardware_tier=self._hardware_tier,
            allow_tf32=self._tf32_enabled,
            matmul_precision=self._matmul_precision,
        )

    @property
    def precision_label(self) -> str:
        return "FP16" if self._precision == "fp16" else "FP32"

    def _prepare_sequence(self, frame: FrameRequest) -> None:
        shape = frame.image.shape[:2]
        reset = False
        if self._last_session_id is not None and frame.session_id != self._last_session_id:
            reset = True
        if self._last_frame_id is not None and frame.frame_id <= self._last_frame_id:
            reset = True
        if self._sequence_shape is not None and shape != self._sequence_shape:
            reset = True
        if reset:
            self._reset_sequence_state(clear_cuda=True)

    def _infer_depth(self, bgr_image: np.ndarray) -> tuple[np.ndarray, float, float]:
        torch = self._torch
        assert torch is not None and self._model is not None and self._device is not None
        original = self._validated_bgr(bgr_image)
        preprocess_started = time.perf_counter()
        rgb = np.ascontiguousarray(original[:, :, ::-1])
        inference_rgb = rgb
        if self._max_resolution > 0 and max(rgb.shape[:2]) > self._max_resolution:
            scale = self._max_resolution / float(max(rgb.shape[:2]))
            resized_h = max(1, int(round(rgb.shape[0] * scale)))
            resized_w = max(1, int(round(rgb.shape[1] * scale)))
            inference_rgb = cv2.resize(rgb, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
        if self._device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            raw = self._model.infer_video_depth_one(
                inference_rgb,
                input_size=self._input_size,
                device=self._device.type,
                fp32=self._precision == "fp32",
            )
            end_event.record()
            torch.cuda.synchronize(self._device)
            latency_ms = float(start_event.elapsed_time(end_event))
        else:
            started = time.perf_counter()
            raw = self._model.infer_video_depth_one(
                inference_rgb,
                input_size=self._input_size,
                device=self._device.type,
                fp32=True,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
        depth = np.squeeze(self._to_numpy(raw))
        if depth.ndim != 2:
            raise RuntimeError(
                f"Expected a 2D Video Depth Anything output, got {tuple(depth.shape)}."
            )
        if depth.shape != original.shape[:2]:
            depth = cv2.resize(depth, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)
        depth = np.clip(depth, 0.0, self._max_depth_m).astype(np.float32, copy=False)
        return np.ascontiguousarray(depth), latency_ms, preprocess_ms

    def _to_geometry_result(
        self,
        frame: FrameRequest,
        depth: np.ndarray,
        latency_ms: float,
        *,
        preprocess_ms: float | None,
        queue_wait_ms: float | None,
        point_budget: int,
    ) -> GeometryResult:
        valid_mask = np.ascontiguousarray(np.isfinite(depth) & (depth > 0.0))
        height, width = depth.shape
        intrinsics = intrinsics_from_virtual_fov(width, height, self._virtual_fov_degrees)
        stride = stride_for_point_budget(height, width, point_budget)
        points = depth_to_points(
            depth[::stride, ::stride],
            intrinsics=scale_intrinsics(intrinsics, 1.0 / float(stride)),
        )
        rgb = np.ascontiguousarray(frame.image[:, :, ::-1])
        allocated, reserved, peak, device_used, total = self._vram_usage()
        return GeometryResult(
            session_id=frame.session_id,
            frame_id=frame.frame_id,
            timestamp=time.time(),
            rgb=frame.image,
            depth=depth,
            points=np.ascontiguousarray(points, dtype=np.float32),
            colors=np.ascontiguousarray(rgb[::stride, ::stride], dtype=np.uint8),
            intrinsics=np.ascontiguousarray(intrinsics, dtype=np.float32),
            valid_mask=valid_mask,
            point_stride=stride,
            latency_ms=latency_ms,
            preprocess_ms=preprocess_ms,
            queue_wait_ms=queue_wait_ms,
            is_metric=True,
            model_id=VIDEO_DEPTH_ANYTHING_MODEL_ID,
            model_name=self._model_name,
            precision=self.precision_label,
            depth_scale="metric",
            intrinsics_source="estimated",
            confidence_kind="none",
            vram_used_gib=reserved,
            vram_allocated_gib=allocated,
            vram_reserved_gib=reserved,
            vram_peak_reserved_gib=peak,
            vram_device_used_gib=device_used,
            vram_total_gib=total,
        )

    def _reset_sequence_state(self, *, clear_cuda: bool) -> bool:
        model = self._model
        had_state = bool(
            self._last_session_id is not None
            or self._last_frame_id is not None
            or self._sequence_shape is not None
            or (model is not None and getattr(model, "id", -1) not in {-1, None})
        )
        if model is not None:
            reset_hook = getattr(model, "reset_stream", None)
            if callable(reset_hook):
                reset_hook()
            else:
                if hasattr(model, "transform"):
                    model.transform = None
                if hasattr(model, "frame_id_list"):
                    model.frame_id_list = []
                if hasattr(model, "frame_cache_list"):
                    model.frame_cache_list = []
                if hasattr(model, "id"):
                    model.id = -1
                for name in ("frame_height", "frame_width"):
                    if hasattr(model, name):
                        try:
                            delattr(model, name)
                        except AttributeError:
                            pass
        self._last_session_id = None
        self._last_frame_id = None
        self._sequence_shape = None
        if had_state:
            self._sequence_resets += 1
        gc.collect()
        if clear_cuda:
            self._clear_cuda_cache()
        return had_state

    def _load_official_model(self, source_dir: str, encoder: str) -> Any:
        source = str(Path(source_dir).resolve())
        # Keep the official checkout first during lazy import. This is especially
        # important for Video Depth Anything, whose upstream code imports a
        # top-level ``utils`` package from its own repository.
        if source in sys.path:
            sys.path.remove(source)
        sys.path.insert(0, source)
        try:
            from video_depth_anything.video_depth_stream import VideoDepthAnything
        except (ImportError, ModuleNotFoundError) as exc:
            raise VideoDepthAnythingDependencyError(
                "Video Depth Anything could not be imported. Run: "
                "python scripts/install_depth_anything_models.py"
            ) from exc
        return VideoDepthAnything(**dict(_MODEL_CONFIGS[encoder]))

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
            raise VideoDepthAnythingCheckpointError(
                "Video Depth Anything checkpoint does not contain a state dictionary."
            )
        return checkpoint

    @staticmethod
    def _validated_bgr(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError("Video Depth Anything expects a NumPy image.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Video Depth Anything expects an H×W×3 BGR image.")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image)

    def _import_torch(self):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise VideoDepthAnythingDependencyError(
                "PyTorch is required for Video Depth Anything."
            ) from exc
        return torch

    @staticmethod
    def _resolve_project_path(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else Path(__file__).resolve().parents[3] / path

    def _require_loaded(self) -> None:
        if not self._loaded or self._model is None:
            raise RuntimeError("VideoDepthAnythingGeometryModel is not loaded.")

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
        before = self._vram_usage()
        self._reset_sequence_state(clear_cuda=False)
        self._loaded = False
        model = self._model
        self._model = None
        if model is not None:
            del model
        gc.collect()
        self._clear_cuda_cache()
        after = self._vram_usage()
        self._device = None
        return {
            "released": True,
            "device": str(device) if device is not None else "",
            "before_allocated_gib": before[0],
            "before_reserved_gib": before[1],
            "before_device_used_gib": before[3],
            "after_allocated_gib": after[0],
            "after_reserved_gib": after[1],
            "after_device_used_gib": after[3],
            "total_gib": after[4],
            "sequence_resets": self._sequence_resets,
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

    def _oom_error(self) -> VideoDepthAnythingOutOfMemoryError:
        return VideoDepthAnythingOutOfMemoryError(
            f"{self._model_name} ran out of GPU memory. Stop inference, close other "
            "GPU workloads, choose Efficient, then restart the worker."
        )
