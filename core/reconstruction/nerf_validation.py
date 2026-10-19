"""Phase 8.7E3 real-scene NeRF validation contracts.

E3 validates appearance quality on held-out real views.  It intentionally does
not make NeRF canonical geometry, metric, or certified-metrology authority and
it does not own VRAM/FPS/robustness hardening (Phase 8.7E4).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.contracts.validation import require_nonempty
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.neural_scene import NeuralSceneDataset, neural_scene_dataset_fingerprint
from core.reconstruction.nerf import NerfBackendKind, NerfOptimizationProfile

NERF_REAL_SCENE_VALIDATION_SCHEMA_VERSION = 1
NERF_HELD_OUT_RENDER_SCHEMA_VERSION = 1
NERF_QUALITY_EVALUATION_SCHEMA_VERSION = 1
NERF_E3_MIN_TRAINING_FRAMES = 8
NERF_E3_MIN_VALIDATION_FRAMES = 3
NERF_E3_DEFAULT_TRAINING_STEPS = 1000
NERF_E3_MIN_TRAINING_STEPS = 128
NERF_E3_MAX_TRAINING_STEPS = 5000


def _sha256(value: object, field_name: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ReconstructionValidationError(f"{field_name} must be a SHA-256 hex digest.")
    return digest


def _finite_nonnegative(value: object, field_name: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed < 0.0:
        raise ReconstructionValidationError(f"{field_name} must be finite and non-negative.")
    return parsed


def _frame_ids(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    result = tuple(require_nonempty(value, field_name=field_name) for value in values)
    if len(set(result)) != len(result):
        raise ReconstructionValidationError(f"{field_name} values must be unique.")
    return result


def deterministic_nerf_validation_split(
    frame_ids: Sequence[str],
    *,
    validation_count: int = NERF_E3_MIN_VALIDATION_FRAMES,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return a deterministic trajectory-spread train/held-out split.

    Held-out views are selected at approximately even interior positions.  This
    avoids choosing only adjacent cameras while keeping the split reproducible.
    """

    ids = _frame_ids(frame_ids, "frame_id")
    count = int(validation_count)
    if count < NERF_E3_MIN_VALIDATION_FRAMES:
        raise ReconstructionValidationError(
            f"E3 requires at least {NERF_E3_MIN_VALIDATION_FRAMES} held-out views."
        )
    if len(ids) - count < NERF_E3_MIN_TRAINING_FRAMES:
        raise ReconstructionValidationError(
            f"E3 requires at least {NERF_E3_MIN_TRAINING_FRAMES} training views after hold-out."
        )

    # Interior quantiles: for N=24,K=4 -> indices near 5,9,14,18.
    raw = [round((index + 1) * (len(ids) - 1) / (count + 1)) for index in range(count)]
    selected_indices: list[int] = []
    for value in raw:
        candidate = max(1, min(len(ids) - 2, int(value)))
        while candidate in selected_indices and candidate < len(ids) - 2:
            candidate += 1
        while candidate in selected_indices and candidate > 1:
            candidate -= 1
        if candidate in selected_indices:
            raise ReconstructionValidationError("Could not create a unique deterministic held-out split.")
        selected_indices.append(candidate)
    selected = set(selected_indices)
    validation = tuple(ids[index] for index in selected_indices)
    training = tuple(value for index, value in enumerate(ids) if index not in selected)
    return training, validation


@dataclass(frozen=True, slots=True)
class NerfQualitySanityPolicy:
    """Conservative non-degeneracy gates, not production quality targets."""

    minimum_psnr_db: float = 8.0
    minimum_ssim: float = 0.05
    maximum_lpips: float = 0.95
    minimum_validation_views: int = NERF_E3_MIN_VALIDATION_FRAMES

    def __post_init__(self) -> None:
        psnr = _finite_nonnegative(self.minimum_psnr_db, "minimum_psnr_db")
        ssim = float(self.minimum_ssim)
        lpips = _finite_nonnegative(self.maximum_lpips, "maximum_lpips")
        views = int(self.minimum_validation_views)
        if not isfinite(ssim) or not 0.0 <= ssim <= 1.0:
            raise ReconstructionValidationError("minimum_ssim must lie in [0, 1].")
        if lpips > 1.5:
            raise ReconstructionValidationError("maximum_lpips is outside the conservative sanity range.")
        if views < NERF_E3_MIN_VALIDATION_FRAMES:
            raise ReconstructionValidationError(
                f"minimum_validation_views must be >= {NERF_E3_MIN_VALIDATION_FRAMES}."
            )
        object.__setattr__(self, "minimum_psnr_db", psnr)
        object.__setattr__(self, "minimum_ssim", ssim)
        object.__setattr__(self, "maximum_lpips", lpips)
        object.__setattr__(self, "minimum_validation_views", views)

    def as_dict(self) -> dict[str, Any]:
        return {
            "minimum_psnr_db": self.minimum_psnr_db,
            "minimum_ssim": self.minimum_ssim,
            "maximum_lpips": self.maximum_lpips,
            "minimum_validation_views": self.minimum_validation_views,
            "scope": "nondegenerate_real_scene_sanity_only",
        }


@dataclass(frozen=True, slots=True)
class NerfRealSceneValidationSpec:
    validation_id: str
    source_label: str
    dataset: NeuralSceneDataset
    training_frame_ids: tuple[str, ...]
    validation_frame_ids: tuple[str, ...]
    optimization_profile: NerfOptimizationProfile = NerfOptimizationProfile.EFFICIENT
    training_steps: int = NERF_E3_DEFAULT_TRAINING_STEPS
    quality_policy: NerfQualitySanityPolicy = NerfQualitySanityPolicy()
    backend_kind: NerfBackendKind = NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL
    schema_version: int = NERF_REAL_SCENE_VALIDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "validation_id", require_nonempty(self.validation_id, field_name="validation_id"))
        object.__setattr__(self, "source_label", require_nonempty(self.source_label, field_name="source_label"))
        if not isinstance(self.dataset, NeuralSceneDataset):
            raise TypeError("dataset must be NeuralSceneDataset.")
        if self.backend_kind is not NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL:
            raise ReconstructionValidationError("E3 requires the E2-qualified real Nerfstudio backend.")
        if not isinstance(self.optimization_profile, NerfOptimizationProfile):
            raise TypeError("optimization_profile must be NerfOptimizationProfile.")
        if not isinstance(self.quality_policy, NerfQualitySanityPolicy):
            raise TypeError("quality_policy must be NerfQualitySanityPolicy.")
        training = _frame_ids(self.training_frame_ids, "training_frame_id")
        validation = _frame_ids(self.validation_frame_ids, "validation_frame_id")
        if len(training) < NERF_E3_MIN_TRAINING_FRAMES:
            raise ReconstructionValidationError(
                f"E3 requires at least {NERF_E3_MIN_TRAINING_FRAMES} training frames."
            )
        if len(validation) < self.quality_policy.minimum_validation_views:
            raise ReconstructionValidationError("E3 does not have enough held-out validation views.")
        if set(training) & set(validation):
            raise ReconstructionValidationError("E3 training and validation splits must be disjoint.")
        dataset_ids = tuple(frame.frame_id for frame in self.dataset.frames)
        if set(training) | set(validation) != set(dataset_ids):
            raise ReconstructionValidationError("E3 train/validation split must cover the dataset exactly.")
        steps = int(self.training_steps)
        if not NERF_E3_MIN_TRAINING_STEPS <= steps <= NERF_E3_MAX_TRAINING_STEPS:
            raise ReconstructionValidationError(
                f"E3 training_steps must lie in [{NERF_E3_MIN_TRAINING_STEPS}, {NERF_E3_MAX_TRAINING_STEPS}]."
            )
        if int(self.schema_version) != NERF_REAL_SCENE_VALIDATION_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported NeRF real-scene validation schema version.")
        object.__setattr__(self, "training_frame_ids", training)
        object.__setattr__(self, "validation_frame_ids", validation)
        object.__setattr__(self, "training_steps", steps)
        object.__setattr__(self, "schema_version", NERF_REAL_SCENE_VALIDATION_SCHEMA_VERSION)

    @property
    def dataset_fingerprint_sha256(self) -> str:
        return neural_scene_dataset_fingerprint(self.dataset)

    @property
    def validation_fingerprint_sha256(self) -> str:
        payload = self.as_dict(include_fingerprint=False)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "validation_id": self.validation_id,
            "source_label": self.source_label,
            "backend_kind": self.backend_kind.value,
            "optimization_profile": self.optimization_profile.value,
            "training_steps": self.training_steps,
            "dataset_fingerprint_sha256": self.dataset_fingerprint_sha256,
            "training_frame_ids": list(self.training_frame_ids),
            "validation_frame_ids": list(self.validation_frame_ids),
            "quality_policy": self.quality_policy.as_dict(),
            "scene_role": "appearance_only",
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
            "performance_hardening_deferred_to": "8.7E4",
        }
        if include_fingerprint:
            payload["validation_fingerprint_sha256"] = self.validation_fingerprint_sha256
        return payload


@dataclass(frozen=True, slots=True)
class NerfHeldOutRender:
    frame_id: str
    reference_image_path: str
    rendered_image_path: str
    width: int
    height: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", require_nonempty(self.frame_id, field_name="frame_id"))
        for field_name in ("reference_image_path", "rendered_image_path"):
            path = Path(require_nonempty(getattr(self, field_name), field_name=field_name)).expanduser()
            if not path.is_absolute():
                raise ReconstructionValidationError(f"{field_name} must be absolute.")
            object.__setattr__(self, field_name, str(path.resolve()))
        for field_name in ("width", "height"):
            value = int(getattr(self, field_name))
            if value < 64:
                raise ReconstructionValidationError(f"{field_name} must be >= 64 for E3 LPIPS evaluation.")
            object.__setattr__(self, field_name, value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "reference_image_path": self.reference_image_path,
            "rendered_image_path": self.rendered_image_path,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NerfHeldOutRender":
        try:
            return cls(
                frame_id=str(payload["frame_id"]),
                reference_image_path=str(payload["reference_image_path"]),
                rendered_image_path=str(payload["rendered_image_path"]),
                width=int(payload["width"]),
                height=int(payload["height"]),
            )
        except (KeyError, TypeError, ValueError, ReconstructionValidationError) as exc:
            raise ReconstructionValidationError(f"Invalid NeRF held-out render: {exc}") from exc


@dataclass(frozen=True, slots=True)
class NerfHeldOutRenderManifest:
    validation_fingerprint_sha256: str
    dataset_fingerprint_sha256: str
    validation_frame_ids: tuple[str, ...]
    renders: tuple[NerfHeldOutRender, ...]
    backend_kind: NerfBackendKind = NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL
    schema_version: int = NERF_HELD_OUT_RENDER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.backend_kind is not NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL:
            raise ReconstructionValidationError("E3 render manifest requires Nerfstudio/Nerfacto.")
        object.__setattr__(self, "validation_fingerprint_sha256", _sha256(self.validation_fingerprint_sha256, "validation_fingerprint_sha256"))
        object.__setattr__(self, "dataset_fingerprint_sha256", _sha256(self.dataset_fingerprint_sha256, "dataset_fingerprint_sha256"))
        ids = _frame_ids(self.validation_frame_ids, "validation_frame_id")
        renders = tuple(self.renders)
        if len(ids) < NERF_E3_MIN_VALIDATION_FRAMES:
            raise ReconstructionValidationError("E3 render manifest requires at least three held-out views.")
        if any(not isinstance(value, NerfHeldOutRender) for value in renders):
            raise TypeError("renders must contain NerfHeldOutRender values.")
        render_ids = tuple(value.frame_id for value in renders)
        if len(render_ids) != len(ids) or set(render_ids) != set(ids):
            raise ReconstructionValidationError("E3 renders must cover each held-out frame exactly once.")
        if int(self.schema_version) != NERF_HELD_OUT_RENDER_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported E3 render manifest schema version.")
        object.__setattr__(self, "validation_frame_ids", ids)
        object.__setattr__(self, "renders", renders)
        object.__setattr__(self, "schema_version", NERF_HELD_OUT_RENDER_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        # benchmark_fingerprint_sha256 is intentionally emitted as an adapter
        # alias so E3 can reuse the already-qualified backend-neutral D3 metrics
        # worker without duplicating PSNR/SSIM/LPIPS implementations.
        return {
            "schema_version": self.schema_version,
            "backend_kind": self.backend_kind.value,
            "validation_fingerprint_sha256": self.validation_fingerprint_sha256,
            "benchmark_fingerprint_sha256": self.validation_fingerprint_sha256,
            "dataset_fingerprint_sha256": self.dataset_fingerprint_sha256,
            "validation_frame_ids": list(self.validation_frame_ids),
            "renders": [value.as_dict() for value in self.renders],
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NerfHeldOutRenderManifest":
        try:
            raw = payload["renders"]
            if not isinstance(raw, list):
                raise ReconstructionValidationError("renders must be a JSON array.")
            fingerprint = payload.get("validation_fingerprint_sha256", payload.get("benchmark_fingerprint_sha256", ""))
            return cls(
                validation_fingerprint_sha256=str(fingerprint),
                dataset_fingerprint_sha256=str(payload["dataset_fingerprint_sha256"]),
                validation_frame_ids=tuple(str(value) for value in payload["validation_frame_ids"]),
                renders=tuple(NerfHeldOutRender.from_payload(value) for value in raw),
                backend_kind=NerfBackendKind(str(payload["backend_kind"])),
                schema_version=int(payload.get("schema_version", 0)),
            )
        except (KeyError, TypeError, ValueError, ReconstructionValidationError) as exc:
            raise ReconstructionValidationError(f"Invalid NeRF held-out render manifest: {exc}") from exc


@dataclass(frozen=True, slots=True)
class NerfHeldOutViewMetrics:
    frame_id: str
    psnr_db: float
    ssim: float
    lpips: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", require_nonempty(self.frame_id, field_name="frame_id"))
        object.__setattr__(self, "psnr_db", _finite_nonnegative(self.psnr_db, "psnr_db"))
        ssim = float(self.ssim)
        if not isfinite(ssim) or not 0.0 <= ssim <= 1.0:
            raise ReconstructionValidationError("ssim must lie in [0, 1].")
        object.__setattr__(self, "ssim", ssim)
        object.__setattr__(self, "lpips", _finite_nonnegative(self.lpips, "lpips"))

    def as_dict(self) -> dict[str, Any]:
        return {"frame_id": self.frame_id, "psnr_db": self.psnr_db, "ssim": self.ssim, "lpips": self.lpips}


@dataclass(frozen=True, slots=True)
class NerfImageQualityMetrics:
    psnr_db: float
    ssim: float
    lpips: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "psnr_db", _finite_nonnegative(self.psnr_db, "psnr_db"))
        ssim = float(self.ssim)
        if not isfinite(ssim) or not 0.0 <= ssim <= 1.0:
            raise ReconstructionValidationError("ssim must lie in [0, 1].")
        object.__setattr__(self, "ssim", ssim)
        object.__setattr__(self, "lpips", _finite_nonnegative(self.lpips, "lpips"))

    def as_dict(self) -> dict[str, Any]:
        return {"measured": True, "psnr_db": self.psnr_db, "ssim": self.ssim, "lpips": self.lpips}


def aggregate_nerf_held_out_metrics(views: Sequence[NerfHeldOutViewMetrics]) -> NerfImageQualityMetrics:
    values = tuple(views)
    if not values or any(not isinstance(value, NerfHeldOutViewMetrics) for value in values):
        raise ReconstructionValidationError("E3 quality aggregation requires held-out view metrics.")
    if len({value.frame_id for value in values}) != len(values):
        raise ReconstructionValidationError("Held-out metric frame IDs must be unique.")
    count = float(len(values))
    return NerfImageQualityMetrics(
        psnr_db=sum(value.psnr_db for value in values) / count,
        ssim=sum(value.ssim for value in values) / count,
        lpips=sum(value.lpips for value in values) / count,
    )


@dataclass(frozen=True, slots=True)
class NerfHeldOutQualityEvaluation:
    validation_fingerprint_sha256: str
    dataset_fingerprint_sha256: str
    validation_frame_ids: tuple[str, ...]
    views: tuple[NerfHeldOutViewMetrics, ...]
    aggregate: NerfImageQualityMetrics
    evaluator_name: str = "depth3d-heldout-quality"
    psnr_definition: str = "rgb_mse_unit_range_mse_floor_1e-12"
    ssim_definition: str = "gaussian_11_sigma_1.5_rgb_unit_range"
    lpips_network: str = "alex"
    lpips_version: str = "0.1"
    lpips_package_version: str = "0.1.4"
    backend_kind: NerfBackendKind = NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL
    schema_version: int = NERF_QUALITY_EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.backend_kind is not NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL:
            raise ReconstructionValidationError("E3 quality evaluation requires Nerfstudio/Nerfacto.")
        object.__setattr__(self, "validation_fingerprint_sha256", _sha256(self.validation_fingerprint_sha256, "validation_fingerprint_sha256"))
        object.__setattr__(self, "dataset_fingerprint_sha256", _sha256(self.dataset_fingerprint_sha256, "dataset_fingerprint_sha256"))
        ids = _frame_ids(self.validation_frame_ids, "validation_frame_id")
        views = tuple(self.views)
        if len(ids) < NERF_E3_MIN_VALIDATION_FRAMES:
            raise ReconstructionValidationError("E3 quality evaluation requires at least three views.")
        if any(not isinstance(value, NerfHeldOutViewMetrics) for value in views):
            raise TypeError("views must contain NerfHeldOutViewMetrics values.")
        if len(views) != len(ids) or set(value.frame_id for value in views) != set(ids):
            raise ReconstructionValidationError("E3 quality metrics must cover every held-out view exactly once.")
        if not isinstance(self.aggregate, NerfImageQualityMetrics):
            raise TypeError("aggregate must be NerfImageQualityMetrics.")
        recomputed = aggregate_nerf_held_out_metrics(views)
        for field_name in ("psnr_db", "ssim", "lpips"):
            if abs(float(getattr(self.aggregate, field_name)) - float(getattr(recomputed, field_name))) > 1e-9:
                raise ReconstructionValidationError("Aggregate E3 quality metrics do not match per-view evidence.")
        for field_name in ("evaluator_name", "psnr_definition", "ssim_definition", "lpips_network", "lpips_version", "lpips_package_version"):
            object.__setattr__(self, field_name, require_nonempty(getattr(self, field_name), field_name=field_name))
        if (self.lpips_network, self.lpips_version, self.lpips_package_version) != ("alex", "0.1", "0.1.4"):
            raise ReconstructionValidationError("E3 reuses the qualified LPIPS AlexNet 0.1 / package 0.1.4 evaluator.")
        if int(self.schema_version) != NERF_QUALITY_EVALUATION_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported E3 quality evaluation schema version.")
        object.__setattr__(self, "validation_frame_ids", ids)
        object.__setattr__(self, "views", views)
        object.__setattr__(self, "schema_version", NERF_QUALITY_EVALUATION_SCHEMA_VERSION)

    @property
    def nondegenerate(self) -> bool:
        return True

    def passes(self, policy: NerfQualitySanityPolicy) -> bool:
        if not isinstance(policy, NerfQualitySanityPolicy):
            raise TypeError("policy must be NerfQualitySanityPolicy.")
        return (
            len(self.views) >= policy.minimum_validation_views
            and self.aggregate.psnr_db >= policy.minimum_psnr_db
            and self.aggregate.ssim >= policy.minimum_ssim
            and self.aggregate.lpips <= policy.maximum_lpips
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend_kind": self.backend_kind.value,
            "validation_fingerprint_sha256": self.validation_fingerprint_sha256,
            "dataset_fingerprint_sha256": self.dataset_fingerprint_sha256,
            "validation_frame_ids": list(self.validation_frame_ids),
            "views": [value.as_dict() for value in self.views],
            "aggregate": self.aggregate.as_dict(),
            "evaluator": {
                "name": self.evaluator_name,
                "psnr_definition": self.psnr_definition,
                "ssim_definition": self.ssim_definition,
                "lpips_network": self.lpips_network,
                "lpips_version": self.lpips_version,
                "lpips_package_version": self.lpips_package_version,
            },
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        }

    @classmethod
    def from_metrics_worker_payload(cls, payload: Mapping[str, Any]) -> "NerfHeldOutQualityEvaluation":
        try:
            raw_views = payload["views"]
            evaluator = payload["evaluator"]
            aggregate = payload["aggregate"]
            if not isinstance(raw_views, list) or not isinstance(evaluator, Mapping) or not isinstance(aggregate, Mapping):
                raise ReconstructionValidationError("Invalid E3 metric worker payload.")
            fingerprint = str(payload.get("validation_fingerprint_sha256", payload.get("benchmark_fingerprint_sha256", "")))
            return cls(
                validation_fingerprint_sha256=fingerprint,
                dataset_fingerprint_sha256=str(payload["dataset_fingerprint_sha256"]),
                validation_frame_ids=tuple(str(value) for value in payload["validation_frame_ids"]),
                views=tuple(
                    NerfHeldOutViewMetrics(
                        frame_id=str(value["frame_id"]),
                        psnr_db=float(value["psnr_db"]),
                        ssim=float(value["ssim"]),
                        lpips=float(value["lpips"]),
                    )
                    for value in raw_views
                ),
                aggregate=NerfImageQualityMetrics(
                    psnr_db=float(aggregate["psnr_db"]),
                    ssim=float(aggregate["ssim"]),
                    lpips=float(aggregate["lpips"]),
                ),
                evaluator_name=str(evaluator["name"]),
                psnr_definition=str(evaluator["psnr_definition"]),
                ssim_definition=str(evaluator["ssim_definition"]),
                lpips_network=str(evaluator["lpips_network"]),
                lpips_version=str(evaluator["lpips_version"]),
                lpips_package_version=str(evaluator["lpips_package_version"]),
                backend_kind=NerfBackendKind(str(payload["backend_kind"])),
            )
        except (KeyError, TypeError, ValueError, ReconstructionValidationError) as exc:
            raise ReconstructionValidationError(f"Invalid E3 quality evaluation: {exc}") from exc


def write_nerf_real_scene_validation_spec(spec: NerfRealSceneValidationSpec, path: str | Path) -> Path:
    if not isinstance(spec, NerfRealSceneValidationSpec):
        raise TypeError("spec must be NerfRealSceneValidationSpec.")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(spec.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def write_nerf_held_out_render_manifest(manifest: NerfHeldOutRenderManifest, path: str | Path) -> Path:
    if not isinstance(manifest, NerfHeldOutRenderManifest):
        raise TypeError("manifest must be NerfHeldOutRenderManifest.")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def read_nerf_held_out_render_manifest(path: str | Path) -> NerfHeldOutRenderManifest:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ReconstructionValidationError(f"E3 render manifest does not exist: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Cannot read E3 render manifest: {exc}") from exc
    return NerfHeldOutRenderManifest.from_payload(payload)


def write_nerf_held_out_quality_evaluation(evaluation: NerfHeldOutQualityEvaluation, path: str | Path) -> Path:
    if not isinstance(evaluation, NerfHeldOutQualityEvaluation):
        raise TypeError("evaluation must be NerfHeldOutQualityEvaluation.")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(evaluation.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
