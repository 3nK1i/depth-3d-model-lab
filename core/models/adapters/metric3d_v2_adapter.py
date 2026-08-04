"""Metric3D v2 adapter for metric depth, normals, and confidence.

The official Metric3D source is loaded lazily from a project-local checkout via
PyTorch Hub. Controllers and the generic worker remain model-agnostic. Because
camera calibration is intentionally postponed, the adapter uses the shared
virtual-FOV estimate for intrinsics and records that provenance explicitly.
"""

from __future__ import annotations

import gc
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
from core.models.catalog import METRIC3D_V2_DISPLAY_LABEL, METRIC3D_V2_MODEL_ID
from core.models.registry import get_model_descriptor
from core.models.result import GeometryResult
from core.runtime.messages import FrameRequest

_DEFAULT_SOURCE_DIR = "third_party/Metric3D"
_DEFAULT_CHECKPOINT = "checkpoints/metric3d-v2/metric_depth_vit_small_800k.pth"
_DEFAULT_ENTRYPOINT = "metric3d_vit_small"
_DEFAULT_MODEL_NAME = "Metric3D v2 ViT-S"
_DEFAULT_INPUT_HEIGHT = 616
_DEFAULT_INPUT_WIDTH = 1064
_DEFAULT_CANONICAL_FOCAL = 1000.0
_DEFAULT_VIRTUAL_FOV = 60.0
_DEFAULT_MAX_DEPTH = 300.0
_DEFAULT_POINT_BUDGET = 250_000
_DEFAULT_WARMUP_SIZE = 256
_DEFAULT_WARMUP_RUNS = 1
_IMAGE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_IMAGE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
_VALID_ENTRYPOINTS = {"metric3d_vit_small", "metric3d_vit_large"}


class Metric3DV2DependencyError(RuntimeError):
    """Raised when the optional official Metric3D runtime is unavailable."""


class Metric3DV2CheckpointError(RuntimeError):
    """Raised when the local source checkout or checkpoint is unavailable."""


class Metric3DV2ConfigurationError(RuntimeError):
    """Raised when a requested runtime configuration is unsupported."""


class Metric3DV2OutOfMemoryError(RuntimeError):
    """Raised with a concise recovery message after a CUDA OOM."""


class Metric3DV2GeometryModel(BaseGeometryModel):
    """Official Metric3D v2 wrapped in the common geometry adapter contract."""

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
        self._entrypoint = _DEFAULT_ENTRYPOINT
        self._model_name = _DEFAULT_MODEL_NAME
        self._input_height = _DEFAULT_INPUT_HEIGHT
        self._input_width = _DEFAULT_INPUT_WIDTH
        self._canonical_focal_length = _DEFAULT_CANONICAL_FOCAL
        self._virtual_fov_degrees = _DEFAULT_VIRTUAL_FOV
        self._max_depth_m = _DEFAULT_MAX_DEPTH
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
            raise Metric3DV2ConfigurationError(
                f"Unsupported Metric3D v2 precision: {requested_precision!r}."
            )
        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise Metric3DV2ConfigurationError(
                f"CUDA device {device!r} was requested, but PyTorch cannot access CUDA."
            )
        if target.type == "cpu" and requested_precision == "fp16":
            requested_precision = "fp32"
        if (
            target.type == "cuda"
            and requested_precision == "fp16"
            and not self._supports_cuda_autocast(torch)
        ):
            requested_precision = "fp32"

        self._device = target
        self._precision = requested_precision
        self._source_dir = str(config.get("source_dir", _DEFAULT_SOURCE_DIR))
        source_dir = self._resolve_project_path(self._source_dir)
        if not (source_dir / "hubconf.py").is_file():
            raise Metric3DV2CheckpointError(
                "Metric3D source checkout is missing or incomplete at "
                f"{source_dir}. Run: python scripts/install_metric3d_v2.py"
            )
        self._resolved_source_dir = str(source_dir)

        self._checkpoint = str(config.get("checkpoint", _DEFAULT_CHECKPOINT))
        checkpoint_path = self._resolve_project_path(self._checkpoint)
        if not checkpoint_path.is_file():
            raise Metric3DV2CheckpointError(
                "Metric3D v2 checkpoint is missing at "
                f"{checkpoint_path}. Run: python scripts/download_metric3d_v2_models.py --variant auto"
            )
        self._resolved_checkpoint = str(checkpoint_path)

        self._entrypoint = str(config.get("entrypoint", _DEFAULT_ENTRYPOINT))
        if self._entrypoint not in _VALID_ENTRYPOINTS:
            raise Metric3DV2ConfigurationError(
                f"Unsupported Metric3D v2 entrypoint: {self._entrypoint!r}."
            )
        self._model_name = str(config.get("model_label", _DEFAULT_MODEL_NAME))
        self._input_height = max(64, int(config.get("input_height", _DEFAULT_INPUT_HEIGHT)))
        self._input_width = max(64, int(config.get("input_width", _DEFAULT_INPUT_WIDTH)))
        self._canonical_focal_length = float(
            config.get("canonical_focal_length", _DEFAULT_CANONICAL_FOCAL)
        )
        if self._canonical_focal_length <= 0.0:
            raise Metric3DV2ConfigurationError(
                "Metric3D canonical_focal_length must be positive."
            )
        self._virtual_fov_degrees = float(
            config.get("virtual_fov_degrees", _DEFAULT_VIRTUAL_FOV)
        )
        if not 1.0 <= self._virtual_fov_degrees < 179.0:
            raise Metric3DV2ConfigurationError(
                "Metric3D virtual_fov_degrees must be in [1, 179)."
            )
        self._max_depth_m = float(config.get("max_depth_m", _DEFAULT_MAX_DEPTH))
        if self._max_depth_m <= 0.0:
            raise Metric3DV2ConfigurationError("Metric3D max_depth_m must be positive.")
        self._point_budget = max(1, int(config.get("point_budget", _DEFAULT_POINT_BUDGET)))
        self._warmup_size = max(64, int(config.get("warmup_size", _DEFAULT_WARMUP_SIZE)))
        self._warmup_runs = min(4, max(1, int(config.get("warmup_runs", _DEFAULT_WARMUP_RUNS))))
        self._allow_tf32 = bool(config.get("allow_tf32", False))
        self._matmul_precision = str(config.get("matmul_precision", "high")).lower()
        if self._matmul_precision not in {"highest", "high", "medium"}:
            raise Metric3DV2ConfigurationError(
                f"Unsupported float32 matmul precision: {self._matmul_precision!r}."
            )
        self._hardware_tier = str(
            config.get("hardware_tier", config.get("profile", "efficient"))
        )
        self._configure_torch_runtime()

        loader = self._model_loader or self._load_official_model
        checkpoint_loader = self._checkpoint_loader or self._load_checkpoint
        try:
            model = loader(str(source_dir), self._entrypoint)
            checkpoint = checkpoint_loader(checkpoint_path)
            state_dict = self._extract_state_dict(checkpoint)
            if not hasattr(model, "load_state_dict"):
                raise TypeError("Metric3D loader returned an object without load_state_dict().")
            model.load_state_dict(state_dict, strict=False)
            # Metric3D's inference pipeline creates/promotes some activations as
            # float32. Converting all model parameters to half therefore causes
            # mixed input/bias convolution failures. Keep weights in float32 and
            # use CUDA autocast for the registry's FP16 profile instead.
            if hasattr(model, "float"):
                model = model.float()
            if hasattr(model, "to"):
                model = model.to(target)
            if hasattr(model, "eval"):
                model.eval()
        except Exception as exc:  # noqa: BLE001
            self._release_model()
            if self._is_cuda_oom(exc):
                raise self._oom_error(exc) from exc
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
                self._infer_model(dummy, point_budget=min(self._point_budget, 4096))
        except Exception as exc:  # noqa: BLE001
            if self._is_cuda_oom(exc):
                raise self._oom_error(exc) from exc
            raise RuntimeError(f"Metric3D v2 warm-up failed: {exc}") from exc

    def infer(self, frame: FrameRequest) -> GeometryResult:
        self._require_loaded()
        worker_started = time.perf_counter()
        queue_wait_ms = self._queue_wait_ms(frame)
        self._reset_peak_memory_stats()
        budget = max(1, int(frame.point_budget or self._point_budget))
        try:
            output, latency_ms, preprocess_ms = self._infer_model(
                frame.image,
                point_budget=budget,
            )
            post_started = time.perf_counter()
            result = self._to_geometry_result(
                frame,
                output,
                latency_ms,
                preprocess_ms=preprocess_ms,
                queue_wait_ms=queue_wait_ms,
                point_budget=budget,
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
        descriptor = get_model_descriptor(METRIC3D_V2_MODEL_ID)
        return descriptor.metadata_with_runtime(
            display_name=self._model_name,
            family=METRIC3D_V2_DISPLAY_LABEL,
            source_dir=self._resolved_source_dir or self._source_dir,
            checkpoint=self._resolved_checkpoint or self._checkpoint,
            entrypoint=self._entrypoint,
            device=str(self._device) if self._device is not None else "",
            precision=self.precision_label,
            input_size=[self._input_height, self._input_width],
            canonical_focal_length=self._canonical_focal_length,
            virtual_fov_degrees=self._virtual_fov_degrees,
            intrinsics_source="estimated",
            metric_scale_status="estimated_camera_intrinsics",
            point_budget=self._point_budget,
            warmup_size=self._warmup_size,
            warmup_runs=self._warmup_runs,
            hardware_tier=self._hardware_tier,
            allow_tf32=self._tf32_enabled,
            matmul_precision=self._matmul_precision,
            load_strategy="local_source_and_checkpoint",
        )

    @property
    def precision_label(self) -> str:
        return "FP16" if self._precision == "fp16" else "FP32"

    def _infer_model(
        self,
        bgr_image: np.ndarray,
        *,
        point_budget: int,
    ) -> tuple[dict[str, Any], float, float]:
        torch = self._torch
        assert torch is not None
        assert self._model is not None
        assert self._device is not None

        preprocess_started = time.perf_counter()
        prepared = self._prepare_input(bgr_image)
        # The official Metric3D example supplies float32 input. FP16 execution
        # is applied with CUDA autocast inside _run_model_inference(), rather
        # than by forcing model parameters or the source tensor to half.
        input_tensor = torch.from_numpy(prepared["normalized_chw"]).unsqueeze(0)
        input_tensor = input_tensor.to(device=self._device, dtype=torch.float32)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0

        if self._device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            raw_output = self._run_model_inference(input_tensor)
            end_event.record()
            torch.cuda.synchronize(self._device)
            latency_ms = float(start_event.elapsed_time(end_event))
        else:
            started = time.perf_counter()
            raw_output = self._run_model_inference(input_tensor)
            latency_ms = (time.perf_counter() - started) * 1000.0

        depth_value, confidence_value, output_dict = self._parse_output(raw_output)
        return {
            "depth": depth_value,
            "confidence": confidence_value,
            "output_dict": output_dict,
            "prepared": prepared,
            "point_budget": point_budget,
        }, latency_ms, preprocess_ms

    def _run_model_inference(self, input_tensor: Any) -> Any:
        torch = self._torch
        assert torch is not None
        assert self._model is not None
        inference_mode = getattr(torch, "inference_mode", None)
        inference_context = (
            inference_mode() if callable(inference_mode) else nullcontext()
        )
        autocast_context = self._cuda_autocast_context()
        with inference_context:
            with autocast_context:
                return self._model.inference({"input": input_tensor})

    @staticmethod
    def _supports_cuda_autocast(torch: Any) -> bool:
        if callable(getattr(torch, "autocast", None)):
            return True
        cuda = getattr(torch, "cuda", None)
        amp = getattr(cuda, "amp", None)
        return callable(getattr(amp, "autocast", None))

    def _cuda_autocast_context(self):
        torch = self._torch
        assert torch is not None
        if (
            self._device is None
            or self._device.type != "cuda"
            or self._precision != "fp16"
        ):
            return nullcontext()

        autocast = getattr(torch, "autocast", None)
        if callable(autocast):
            return autocast(device_type="cuda", dtype=torch.float16)

        cuda = getattr(torch, "cuda", None)
        amp = getattr(cuda, "amp", None)
        legacy_autocast = getattr(amp, "autocast", None)
        if callable(legacy_autocast):
            return legacy_autocast(dtype=torch.float16)

        # load() downgrades to FP32 when neither API is available. This branch
        # is defensive in case a custom torch shim changes after loading.
        return nullcontext()

    @staticmethod
    def _parse_output(raw_output: Any) -> tuple[Any, Any | None, Mapping[str, Any]]:
        if not isinstance(raw_output, (tuple, list)) or len(raw_output) < 3:
            raise RuntimeError(
                "Metric3D v2 returned an unexpected output; expected "
                "(depth, confidence, output_dict)."
            )
        output_dict = raw_output[2]
        if output_dict is None:
            output_dict = {}
        if not isinstance(output_dict, Mapping):
            raise RuntimeError("Metric3D v2 output_dict must be a mapping.")
        return raw_output[0], raw_output[1], output_dict

    def _prepare_input(self, bgr_image: np.ndarray) -> dict[str, Any]:
        rgb = self._validated_rgb(bgr_image)
        height, width = rgb.shape[:2]
        intrinsics = intrinsics_from_virtual_fov(
            width,
            height,
            self._virtual_fov_degrees,
        )
        scale = min(self._input_height / height, self._input_width / width)
        resized_height = max(1, min(self._input_height, int(round(height * scale))))
        resized_width = max(1, min(self._input_width, int(round(width * scale))))
        resized = cv2.resize(
            rgb,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        pad_height = self._input_height - resized_height
        pad_width = self._input_width - resized_width
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=tuple(float(v) for v in _IMAGE_MEAN),
        )
        normalized = (padded.astype(np.float32) - _IMAGE_MEAN) / _IMAGE_STD
        normalized_chw = np.ascontiguousarray(normalized.transpose(2, 0, 1))
        scaled_intrinsics = scale_intrinsics(intrinsics, scale)
        return {
            "normalized_chw": normalized_chw,
            "rgb": rgb,
            "intrinsics": intrinsics,
            "scaled_intrinsics": scaled_intrinsics,
            "scale": float(scale),
            "pad": (pad_top, pad_bottom, pad_left, pad_right),
            "original_shape": (height, width),
        }

    @staticmethod
    def _validated_rgb(bgr_image: np.ndarray) -> np.ndarray:
        if not isinstance(bgr_image, np.ndarray):
            raise TypeError("Metric3D v2 expects a NumPy image.")
        if bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
            raise ValueError("Metric3D v2 expects an H×W×3 BGR image.")
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
        point_budget: int,
    ) -> GeometryResult:
        prepared = output["prepared"]
        depth = self._postprocess_scalar_map(
            output.get("depth"),
            prepared,
            name="depth",
        )
        canonical_to_real_scale = (
            float(prepared["scaled_intrinsics"][0, 0]) / self._canonical_focal_length
        )
        depth = np.ascontiguousarray(depth * canonical_to_real_scale, dtype=np.float32)
        depth = np.clip(depth, 0.0, self._max_depth_m).astype(np.float32, copy=False)

        confidence = None
        if output.get("confidence") is not None:
            confidence = self._postprocess_scalar_map(
                output.get("confidence"),
                prepared,
                name="confidence",
            )
            confidence = np.ascontiguousarray(confidence, dtype=np.float32)

        normals = None
        output_dict = output.get("output_dict") or {}
        normal_value = output_dict.get("prediction_normal")
        if normal_value is not None:
            normals = self._postprocess_normals(normal_value, prepared)

        valid_mask = np.ascontiguousarray(
            np.isfinite(depth) & (depth > 0.0) & (depth <= self._max_depth_m)
        )
        intrinsics = np.ascontiguousarray(prepared["intrinsics"], dtype=np.float32)
        stride = stride_for_point_budget(depth.shape[0], depth.shape[1], point_budget)
        sampled_depth = depth[::stride, ::stride]
        sampled_intrinsics = scale_intrinsics(intrinsics, 1.0 / float(stride))
        points = np.ascontiguousarray(
            depth_to_points(sampled_depth, intrinsics=sampled_intrinsics),
            dtype=np.float32,
        )
        sampled_normals = (
            None
            if normals is None
            else np.ascontiguousarray(normals[::stride, ::stride], dtype=np.float32)
        )
        colors = np.ascontiguousarray(
            prepared["rgb"][::stride, ::stride],
            dtype=np.uint8,
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
            points=points,
            colors=colors,
            normals=sampled_normals,
            intrinsics=intrinsics,
            confidence=confidence,
            valid_mask=valid_mask,
            point_stride=stride,
            latency_ms=latency_ms,
            preprocess_ms=preprocess_ms,
            queue_wait_ms=queue_wait_ms,
            is_metric=True,
            model_id=METRIC3D_V2_MODEL_ID,
            model_name=self._model_name,
            precision=self.precision_label,
            depth_scale="metric",
            intrinsics_source="estimated",
            confidence_kind="confidence" if confidence is not None else "none",
            vram_used_gib=reserved_gib,
            vram_allocated_gib=allocated_gib,
            vram_reserved_gib=reserved_gib,
            vram_peak_reserved_gib=peak_reserved_gib,
            vram_device_used_gib=device_used_gib,
            vram_total_gib=total_gib,
        )

    def _postprocess_scalar_map(
        self,
        value: Any,
        prepared: Mapping[str, Any],
        *,
        name: str,
    ) -> np.ndarray:
        if value is None:
            raise RuntimeError(f"Metric3D v2 output did not contain {name}.")
        array = np.squeeze(self._to_numpy(value, dtype=np.float32))
        if array.ndim != 2:
            raise RuntimeError(
                f"Expected a 2D Metric3D v2 {name} map, got {tuple(array.shape)}."
            )
        cropped = self._crop_padding(array, prepared["pad"])
        height, width = prepared["original_shape"]
        return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

    def _postprocess_normals(
        self,
        value: Any,
        prepared: Mapping[str, Any],
    ) -> np.ndarray:
        array = self._to_numpy(value, dtype=np.float32)
        while array.ndim > 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3:
            raise RuntimeError(
                f"Expected Metric3D v2 normals with three dimensions, got {array.shape}."
            )
        if array.shape[0] in {3, 4}:
            array = array[:3].transpose(1, 2, 0)
        elif array.shape[2] in {3, 4}:
            array = array[:, :, :3]
        else:
            raise RuntimeError("Metric3D v2 normal output does not contain three channels.")
        cropped = self._crop_padding(array, prepared["pad"])
        height, width = prepared["original_shape"]
        resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
        lengths = np.linalg.norm(resized, axis=2, keepdims=True)
        valid = lengths > 1e-6
        normalized = np.zeros_like(resized, dtype=np.float32)
        np.divide(resized, lengths, out=normalized, where=valid)
        return np.ascontiguousarray(normalized, dtype=np.float32)

    @staticmethod
    def _crop_padding(array: np.ndarray, pad: tuple[int, int, int, int]) -> np.ndarray:
        top, bottom, left, right = (int(value) for value in pad)
        height_end = array.shape[0] - bottom if bottom else array.shape[0]
        width_end = array.shape[1] - right if right else array.shape[1]
        cropped = array[top:height_end, left:width_end]
        if cropped.size == 0:
            raise RuntimeError("Metric3D v2 padding removal produced an empty output.")
        return cropped

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
            raise Metric3DV2DependencyError(
                "PyTorch is required for Metric3D v2. Install a compatible CUDA build first."
            ) from exc
        return torch

    def _load_official_model(self, source_dir: str, entrypoint: str) -> Any:
        torch = self._torch
        assert torch is not None
        try:
            return torch.hub.load(
                source_dir,
                entrypoint,
                source="local",
                pretrain=False,
                trust_repo=True,
            )
        except TypeError:
            return torch.hub.load(
                source_dir,
                entrypoint,
                source="local",
                pretrain=False,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise Metric3DV2DependencyError(
                "Metric3D v2 dependencies are missing. Run: "
                "python scripts/install_metric3d_v2.py"
            ) from exc

    def _load_checkpoint(self, path: Path) -> Any:
        torch = self._torch
        assert torch is not None
        try:
            return torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(str(path), map_location="cpu")

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> Mapping[str, Any]:
        if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        if not isinstance(state_dict, Mapping):
            raise Metric3DV2CheckpointError(
                "Metric3D v2 checkpoint does not contain a valid model state dictionary."
            )
        return state_dict

    @staticmethod
    def _resolve_project_path(value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[3] / path

    def _require_loaded(self) -> None:
        if not self._loaded or self._model is None:
            raise RuntimeError("Metric3DV2GeometryModel is not loaded.")

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
            except Exception:  # noqa: BLE001
                pass
        self._loaded = False
        model = self._model
        self._model = None
        if model is not None:
            del model
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

    def _oom_error(self, _exc: BaseException) -> Metric3DV2OutOfMemoryError:
        return Metric3DV2OutOfMemoryError(
            f"{self._model_name} ran out of GPU memory. Stop inference, close other "
            "GPU workloads, choose Efficient, then restart the worker."
        )
