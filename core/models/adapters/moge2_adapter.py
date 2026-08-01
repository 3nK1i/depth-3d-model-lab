"""MoGe-2 adapter for metric monocular geometry inference.

Heavy dependencies are imported lazily inside :meth:`load`, so importing the
application UI never imports MoGe or initializes CUDA. The adapter runs only in
the spawned inference worker.
"""

from __future__ import annotations

import gc
import time
from typing import Any

import numpy as np

from core.geometry.sampling import stride_for_point_budget
from core.models.base import BaseGeometryModel
from core.models.catalog import MOGE2_DISPLAY_LABEL, MOGE2_MODEL_ID
from core.models.result import GeometryResult
from core.runtime.messages import FrameRequest

_DEFAULT_PRETRAINED = "Ruicheng/moge-2-vits-normal"
_DEFAULT_NUM_TOKENS = 1200
_DEFAULT_RESOLUTION_LEVEL = 5
_DEFAULT_POINT_BUDGET = 250_000
_DEFAULT_WARMUP_SIZE = 256


class MoGeDependencyError(RuntimeError):
    """Raised when the optional official MoGe package is unavailable."""


class MoGeConfigurationError(RuntimeError):
    """Raised when the requested runtime configuration is unsupported."""


class MoGeOutOfMemoryError(RuntimeError):
    """Raised with a concise recovery message after a CUDA OOM."""


class Moge2GeometryModel(BaseGeometryModel):
    """Official MoGe-2 ``MoGeModel`` wrapped in the app's adapter contract."""

    def __init__(
        self,
        *,
        model_class: type | None = None,
        torch_module: Any | None = None,
    ) -> None:
        # Optional injection keeps unit tests independent of network downloads.
        self._model_class = model_class
        self._torch = torch_module
        self._model: Any | None = None
        self._device: Any | None = None
        self._loaded = False
        self._precision = "fp32"
        self._use_fp16 = False
        self._pretrained = _DEFAULT_PRETRAINED
        self._model_name = MOGE2_DISPLAY_LABEL
        self._num_tokens = _DEFAULT_NUM_TOKENS
        self._resolution_level = _DEFAULT_RESOLUTION_LEVEL
        self._point_budget = _DEFAULT_POINT_BUDGET
        self._warmup_size = _DEFAULT_WARMUP_SIZE
        self._local_files_only = False

    def load(self, device: str, precision: str, config: dict[str, Any]) -> None:
        torch = self._torch or self._import_torch()
        model_class = self._model_class or self._import_model_class()
        self._torch = torch
        self._model_class = model_class

        requested_precision = str(precision).lower()
        if requested_precision not in {"fp32", "fp16", "bf16"}:
            raise MoGeConfigurationError(
                f"Unsupported MoGe-2 precision: {requested_precision!r}."
            )

        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise MoGeConfigurationError(
                f"CUDA device {device!r} was requested, but PyTorch cannot access CUDA."
            )
        if target.type == "cpu" and requested_precision == "fp16":
            requested_precision = "fp32"

        self._device = target
        self._precision = requested_precision
        self._use_fp16 = requested_precision == "fp16" and target.type == "cuda"
        self._pretrained = str(config.get("pretrained", _DEFAULT_PRETRAINED))
        self._model_name = str(config.get("model_label", MOGE2_DISPLAY_LABEL))
        self._num_tokens = max(1, int(config.get("num_tokens", _DEFAULT_NUM_TOKENS)))
        self._resolution_level = min(
            9,
            max(0, int(config.get("resolution_level", _DEFAULT_RESOLUTION_LEVEL))),
        )
        self._point_budget = max(
            1,
            int(config.get("point_budget", _DEFAULT_POINT_BUDGET)),
        )
        self._warmup_size = max(
            64,
            int(config.get("warmup_size", _DEFAULT_WARMUP_SIZE)),
        )
        self._local_files_only = bool(config.get("local_files_only", False))

        hf_kwargs: dict[str, Any] = {"token": False}
        if self._local_files_only:
            hf_kwargs["local_files_only"] = True

        try:
            model = model_class.from_pretrained(self._pretrained, **hf_kwargs)
            model = model.to(self._device)
            if self._precision == "bf16":
                if self._device.type != "cuda":
                    raise MoGeConfigurationError("BF16 is enabled only for CUDA in this app.")
                model = model.to(dtype=torch.bfloat16)
            model.eval()
            # Native SDPA is an official optional optimization. Keep loading
            # robust on older MoGe versions that do not expose the method.
            if hasattr(model, "enable_pytorch_native_sdpa"):
                model.enable_pytorch_native_sdpa()
        except Exception as exc:  # noqa: BLE001 - enrich model/download errors
            self._release_model()
            if self._is_cuda_oom(exc):
                raise self._oom_error(exc) from exc
            raise RuntimeError(
                f"Failed to load {self._model_name} from {self._pretrained!r}: {exc}"
            ) from exc

        self._model = model
        self._loaded = True

    def warmup(self) -> None:
        self._require_loaded()
        dummy = np.zeros(
            (self._warmup_size, self._warmup_size, 3),
            dtype=np.uint8,
        )
        try:
            self._infer_model(dummy)
        except Exception as exc:  # noqa: BLE001
            if self._is_cuda_oom(exc):
                raise self._oom_error(exc) from exc
            raise RuntimeError(f"MoGe-2 warm-up failed: {exc}") from exc

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
            return GeometryResult(
                session_id=result.session_id,
                frame_id=result.frame_id,
                timestamp=result.timestamp,
                rgb=result.rgb,
                depth=result.depth,
                points=result.points,
                colors=result.colors,
                normals=result.normals,
                intrinsics=result.intrinsics,
                confidence=result.confidence,
                valid_mask=result.valid_mask,
                point_stride=result.point_stride,
                latency_ms=result.latency_ms,
                preprocess_ms=preprocess_ms,
                postprocess_ms=postprocess_ms,
                worker_total_ms=worker_total_ms,
                queue_wait_ms=queue_wait_ms,
                is_metric=result.is_metric,
                model_id=result.model_id,
                model_name=result.model_name,
                precision=result.precision,
                vram_used_gib=result.vram_used_gib,
                vram_allocated_gib=result.vram_allocated_gib,
                vram_reserved_gib=result.vram_reserved_gib,
                vram_peak_reserved_gib=result.vram_peak_reserved_gib,
                vram_device_used_gib=result.vram_device_used_gib,
                vram_total_gib=result.vram_total_gib,
            )
        except Exception as exc:  # noqa: BLE001
            if self._is_cuda_oom(exc):
                self._clear_cuda_cache()
                raise self._oom_error(exc) from exc
            raise

    def unload(self) -> None:
        self._release_model()

    def get_metadata(self) -> dict[str, Any]:
        return {
            "id": MOGE2_MODEL_ID,
            "display_name": self._model_name,
            "family": MOGE2_DISPLAY_LABEL,
            "pretrained": self._pretrained,
            "device": str(self._device) if self._device is not None else "",
            "precision": self.precision_label,
            "num_tokens": self._num_tokens,
            "resolution_level": self._resolution_level,
            "is_metric": True,
            "outputs": {
                "depth": True,
                "points": True,
                "normals": True,
                "intrinsics": True,
                "valid_mask": True,
                "confidence": False,
            },
            "supports_video": True,
        }

    @property
    def precision_label(self) -> str:
        return {
            "fp16": "FP16",
            "bf16": "BF16",
            "fp32": "FP32",
        }[self._precision]

    # -- Internal model execution -----------------------------------------

    def _infer_model(
        self,
        bgr_image: np.ndarray,
    ) -> tuple[dict[str, Any], float, float]:
        torch = self._torch
        assert torch is not None
        assert self._model is not None
        assert self._device is not None

        preprocess_started = time.perf_counter()
        image = self._image_to_tensor(bgr_image)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
        kwargs = {
            "num_tokens": self._num_tokens,
            "resolution_level": self._resolution_level,
            "force_projection": True,
            "apply_mask": False,
            "use_fp16": self._use_fp16,
        }

        if self._device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output = self._model.infer(image, **kwargs)
            end_event.record()
            torch.cuda.synchronize(self._device)
            latency_ms = float(start_event.elapsed_time(end_event))
        else:
            started = time.perf_counter()
            output = self._model.infer(image, **kwargs)
            latency_ms = (time.perf_counter() - started) * 1000.0

        if not isinstance(output, dict):
            raise RuntimeError("MoGe-2 returned an unexpected non-dictionary output.")
        return output, latency_ms, preprocess_ms

    def _image_to_tensor(self, bgr_image: np.ndarray):
        if bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
            raise ValueError("MoGe-2 expects an H×W×3 BGR image.")
        if bgr_image.dtype != np.uint8:
            bgr_image = np.clip(bgr_image, 0, 255).astype(np.uint8)

        torch = self._torch
        assert torch is not None
        assert self._device is not None

        rgb = np.ascontiguousarray(bgr_image[:, :, ::-1])
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        return tensor.to(device=self._device, dtype=torch.float32).div_(255.0)

    def _to_geometry_result(
        self,
        frame: FrameRequest,
        output: dict[str, Any],
        latency_ms: float,
        *,
        preprocess_ms: float | None = None,
        queue_wait_ms: float | None = None,
    ) -> GeometryResult:
        torch = self._torch
        assert torch is not None

        depth_tensor = output.get("depth")
        if depth_tensor is None:
            raise RuntimeError("MoGe-2 output did not contain a depth map.")
        if depth_tensor.ndim != 2:
            raise RuntimeError(
                f"Expected a 2D MoGe-2 depth map, got shape {tuple(depth_tensor.shape)}."
            )

        height, width = map(int, depth_tensor.shape)
        requested_budget = frame.point_budget or self._point_budget
        stride = stride_for_point_budget(height, width, max(1, int(requested_budget)))

        depth = self._tensor_to_numpy(depth_tensor, dtype=np.float32)

        mask_tensor = output.get("mask")
        if mask_tensor is not None:
            valid_mask = self._tensor_to_numpy(mask_tensor, dtype=np.bool_)
        else:
            valid_mask = np.isfinite(depth) & (depth > 0)

        points = None
        points_tensor = output.get("points")
        if points_tensor is not None:
            points = self._tensor_to_numpy(
                points_tensor[::stride, ::stride],
                dtype=np.float32,
            )

        normals = None
        normals_tensor = output.get("normal")
        if normals_tensor is not None:
            normals = self._tensor_to_numpy(
                normals_tensor[::stride, ::stride],
                dtype=np.float32,
            )

        intrinsics = None
        intrinsics_tensor = output.get("intrinsics")
        if intrinsics_tensor is not None:
            intrinsics = self._tensor_to_numpy(intrinsics_tensor, dtype=np.float32)

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
            normals=normals,
            intrinsics=intrinsics,
            valid_mask=valid_mask,
            point_stride=stride,
            latency_ms=latency_ms,
            preprocess_ms=preprocess_ms,
            queue_wait_ms=queue_wait_ms,
            is_metric=True,
            model_id=MOGE2_MODEL_ID,
            model_name=self._model_name,
            precision=self.precision_label,
            vram_used_gib=reserved_gib,
            vram_allocated_gib=allocated_gib,
            vram_reserved_gib=reserved_gib,
            vram_peak_reserved_gib=peak_reserved_gib,
            vram_device_used_gib=device_used_gib,
            vram_total_gib=total_gib,
        )

    def _tensor_to_numpy(self, tensor: Any, *, dtype: np.dtype) -> np.ndarray:
        torch = self._torch
        assert torch is not None
        if not isinstance(tensor, torch.Tensor):
            array = np.asarray(tensor)
        else:
            array = tensor.detach().contiguous().cpu().numpy()
        return np.ascontiguousarray(array.astype(dtype, copy=False))

    # -- Dependency/resource helpers --------------------------------------

    def _import_torch(self):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - base app already detects torch
            raise MoGeDependencyError(
                "PyTorch is required for MoGe-2. Install the model requirements first."
            ) from exc
        return torch

    def _import_model_class(self):
        try:
            from moge.model.v2 import MoGeModel
        except ImportError as exc:
            raise MoGeDependencyError(
                "The official MoGe package is not installed. Run: "
                "python -m pip install -r requirements/models-moge2.txt"
            ) from exc
        return MoGeModel

    def _require_loaded(self) -> None:
        if not self._loaded or self._model is None:
            raise RuntimeError("Moge2GeometryModel is not loaded.")

    def _release_model(self) -> None:
        self._loaded = False
        self._model = None
        gc.collect()
        self._clear_cuda_cache()

    def _clear_cuda_cache(self) -> None:
        torch = self._torch
        if torch is None or not hasattr(torch, "cuda"):
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001 - cleanup must never mask the real error
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
        """Return allocated, reserved, peak-reserved, device-used, total GiB."""
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
        # Cross-process monotonic clocks are shared on supported desktop
        # platforms, but discard clearly invalid values rather than display them.
        if elapsed < 0.0 or elapsed > 60_000.0:
            return None
        return elapsed

    def _is_cuda_oom(self, exc: BaseException) -> bool:
        torch = self._torch
        oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", ())
        return bool(oom_type) and isinstance(exc, oom_type)

    def _oom_error(self, _exc: BaseException) -> MoGeOutOfMemoryError:
        return MoGeOutOfMemoryError(
            f"{self._model_name} ran out of GPU memory. Stop inference, choose "
            "the Efficient profile or a lower point budget, then restart the worker."
        )
