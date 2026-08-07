"""Typed backend capability declarations and adapter interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

from core.contracts.validation import require_nonempty
from core.reconstruction.contracts import InputSourceKind
from core.reconstruction.control import ReconstructionOperationControl
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.events import ReconstructionEvent, ReconstructionStage
from core.reconstruction.profiles import ReconstructionQualityProfile

if TYPE_CHECKING:
    from core.reconstruction.live_contracts import (
        LiveSlamFrame,
        LiveSlamObservation,
        LiveSlamRequest,
    )
    from core.reconstruction.sfm_contracts import OfflineSfmRequest, OfflineSfmResult


class ReconstructionBackendKind(str, Enum):
    OFFLINE_SFM = "offline_sfm"
    LIVE_SLAM = "live_slam"
    DENSE_FUSION = "dense_fusion"
    MESH_GENERATION = "mesh_generation"


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    backend_id: str
    display_name: str
    backend_version: str
    kinds: tuple[ReconstructionBackendKind, ...]
    source_kinds: tuple[InputSourceKind, ...]
    quality_profiles: tuple[ReconstructionQualityProfile, ...]
    supports_cancellation: bool = True
    supports_pause: bool = False
    supports_checkpoint: bool = False
    supports_resume: bool = False
    supports_metric_scale: bool = False
    supports_loop_closure: bool = False
    supports_cpu: bool = True
    requires_cuda: bool = False
    minimum_cpu_memory_bytes: int = 0
    minimum_gpu_memory_bytes: int = 0
    optional_dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "backend_id", require_nonempty(self.backend_id, field_name="backend_id")
        )
        object.__setattr__(
            self, "display_name", require_nonempty(self.display_name, field_name="display_name")
        )
        object.__setattr__(
            self,
            "backend_version",
            require_nonempty(self.backend_version, field_name="backend_version"),
        )
        kinds = tuple(self.kinds)
        sources = tuple(self.source_kinds)
        profiles = tuple(self.quality_profiles)
        if not kinds or any(not isinstance(item, ReconstructionBackendKind) for item in kinds):
            raise ReconstructionValidationError(
                "Backend capabilities require at least one valid backend kind."
            )
        if not sources or any(not isinstance(item, InputSourceKind) for item in sources):
            raise ReconstructionValidationError(
                "Backend capabilities require at least one valid source kind."
            )
        if not profiles or any(
            not isinstance(item, ReconstructionQualityProfile) for item in profiles
        ):
            raise ReconstructionValidationError(
                "Backend capabilities require at least one quality profile."
            )
        if len(set(kinds)) != len(kinds):
            raise ReconstructionValidationError("Backend kinds must be unique.")
        if len(set(sources)) != len(sources):
            raise ReconstructionValidationError("Backend source kinds must be unique.")
        if len(set(profiles)) != len(profiles):
            raise ReconstructionValidationError("Backend quality profiles must be unique.")
        if self.supports_resume and not self.supports_checkpoint:
            raise ReconstructionValidationError(
                "Resume support requires checkpoint support."
            )
        if self.requires_cuda and self.supports_cpu:
            raise ReconstructionValidationError(
                "A CUDA-required backend cannot simultaneously declare CPU support."
            )
        for field_name in ("minimum_cpu_memory_bytes", "minimum_gpu_memory_bytes"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or int(value) < 0:
                raise ReconstructionValidationError(f"{field_name} must not be negative.")
            object.__setattr__(self, field_name, int(value))
        dependencies = tuple(
            require_nonempty(item, field_name="optional_dependency")
            for item in self.optional_dependencies
        )
        if len(set(dependencies)) != len(dependencies):
            raise ReconstructionValidationError("Optional dependencies must be unique.")
        object.__setattr__(self, "kinds", kinds)
        object.__setattr__(self, "source_kinds", sources)
        object.__setattr__(self, "quality_profiles", profiles)
        object.__setattr__(self, "optional_dependencies", dependencies)

    def supports_kind(self, kind: ReconstructionBackendKind) -> bool:
        return kind in self.kinds


@dataclass(frozen=True, slots=True)
class BackendOperationContext:
    session_id: str
    stage: ReconstructionStage
    profile: ReconstructionQualityProfile
    control: ReconstructionOperationControl
    emit_event: Callable[[ReconstructionEvent], None]


@runtime_checkable
class ReconstructionBackend(Protocol):
    @property
    def capabilities(self) -> BackendCapabilities: ...

    def prepare(self, context: BackendOperationContext) -> None: ...

    def shutdown(self) -> None: ...


@runtime_checkable
class OfflineSfmBackend(ReconstructionBackend, Protocol):
    def run_sparse_reconstruction(
        self,
        request: "OfflineSfmRequest",
        context: BackendOperationContext,
    ) -> "OfflineSfmResult": ...


@runtime_checkable
class LiveSlamBackend(ReconstructionBackend, Protocol):
    def start_live_session(
        self,
        request: "LiveSlamRequest",
        context: BackendOperationContext,
    ) -> None: ...

    def track_live_frame(
        self,
        frame: "LiveSlamFrame",
        context: BackendOperationContext,
    ) -> "LiveSlamObservation": ...

    def stop_live_session(self, context: BackendOperationContext) -> None: ...


@runtime_checkable
class DenseFusionBackend(ReconstructionBackend, Protocol):
    def integrate_keyframe(self, keyframe: object, context: BackendOperationContext) -> object: ...


@runtime_checkable
class MeshGenerationBackend(ReconstructionBackend, Protocol):
    def generate_mesh(self, dense_artifact: object, context: BackendOperationContext) -> object: ...
