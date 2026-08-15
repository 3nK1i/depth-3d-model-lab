"""Phase 9.4E dependency-light interactive Gaussian presentation.

The production application reads an integrity-bound persisted ``scene.pt`` on CPU
and visualizes Gaussian means/colors/scales through VisPy.  No gsplat/Graphdeco
runtime is imported or started here.  This is an interactive appearance viewer,
not geometry, scale, measurement, QC, or certified-metrology authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.contracts.validation import metadata_as_dict
from core.reconstruction.persistent_scene import (
    PersistentReconstructionScene,
    verify_persistent_scene_file_references,
)

MAX_INTERACTIVE_GAUSSIANS = 200_000


def _display_indices(total: int, maximum: int) -> np.ndarray:
    total = int(total)
    maximum = max(1, int(maximum))
    if total <= maximum:
        return np.arange(total, dtype=np.int64)
    # Stable, endpoint-preserving sampling. Appearance remains presentation only.
    return np.linspace(0, total - 1, num=maximum, dtype=np.int64)


def _rgba(colors: np.ndarray, opacities: np.ndarray) -> np.ndarray:
    rgb = np.asarray(colors, dtype=np.float32)
    alpha = np.asarray(opacities, dtype=np.float32).reshape(-1)
    if rgb.ndim != 2 or rgb.shape[1] != 3 or alpha.shape != (rgb.shape[0],):
        raise ValueError("Persisted Gaussian colors/opacities have invalid shapes.")
    if not np.isfinite(rgb).all() or not np.isfinite(alpha).all():
        raise ValueError("Persisted Gaussian colors/opacities must be finite.")
    result = np.empty((rgb.shape[0], 4), dtype=np.float32)
    result[:, :3] = np.clip(rgb, 0.0, 1.0)
    # Keep low-opacity splats visible enough for an interactive point-splat view.
    result[:, 3] = np.clip(alpha, 0.08, 1.0)
    return result


def _sizes(scales: np.ndarray) -> np.ndarray:
    values = np.asarray(scales, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Persisted Gaussian scales must be finite (N, 3).")
    magnitudes = np.cbrt(np.maximum(values.prod(axis=1), 1e-18))
    positive = magnitudes[magnitudes > 0.0]
    median = float(np.median(positive)) if positive.size else 1.0
    ratio = np.maximum(magnitudes / max(median, 1e-12), 1e-4)
    # Screen-space point sprites are an interactive approximation of the persisted
    # Gaussian representation; the backend-authored static preview remains the
    # high-fidelity rasterization reference.
    return np.clip(5.0 * np.sqrt(ratio), 2.0, 14.0).astype(np.float32)


@dataclass(frozen=True, slots=True)
class InteractiveGaussianScene:
    scene_id: str
    artifact_id: str
    backend_id: str
    backend_version: str
    scene_path: str
    total_gaussian_count: int
    display_gaussian_count: int
    positions_reconstruction: np.ndarray
    colors_rgba: np.ndarray
    sizes_px: np.ndarray
    metric_geometry_status: str
    canonical_geometry_authoritative: bool = False
    metric_scale_authoritative: bool = False
    measurement_authoritative: bool = False
    qc_authoritative: bool = False
    certified_metrology: bool = False

    @property
    def state_text(self) -> str:
        suffix = "" if self.total_gaussian_count == self.display_gaussian_count else (
            f" • displaying {self.display_gaussian_count:,}/{self.total_gaussian_count:,}"
        )
        return f"Interactive Gaussian appearance ready{suffix}"

    @property
    def authority_text(self) -> str:
        return "Appearance only • picking/measurement remain canonical-geometry based"


def load_interactive_gaussian_scene(
    scene: PersistentReconstructionScene,
    *,
    max_gaussians: int = MAX_INTERACTIVE_GAUSSIANS,
) -> InteractiveGaussianScene:
    """Load a persisted Gaussian scene safely on CPU without a Gaussian runtime."""

    if not isinstance(scene, PersistentReconstructionScene):
        raise TypeError("scene must be PersistentReconstructionScene.")
    artifact = scene.gaussian_artifact
    if artifact is None:
        raise ValueError("Persistent scene has no Gaussian appearance artifact.")
    if str(getattr(artifact.authority, "value", artifact.authority)) != "appearance_only":
        raise ValueError("Gaussian interactive viewer refuses non-appearance authority.")

    snapshot = metadata_as_dict(artifact.payload_snapshot)  # type: ignore[arg-type]
    scene_path_text = str(snapshot.get("scene_path", "")).strip()
    if not scene_path_text:
        raise ValueError("Persisted Gaussian artifact omits scene_path.")
    scene_path = Path(scene_path_text).expanduser().resolve()

    # Verify every persistent file reference before reading scene.pt. This catches
    # mutation/tampering and preserves the existing persistence authority contract.
    refs = verify_persistent_scene_file_references(scene)
    scene_refs = [
        item for item in refs
        if item.artifact_id == artifact.artifact_id and item.role == "scene_path"
    ]
    if len(scene_refs) != 1 or Path(scene_refs[0].path).resolve() != scene_path:
        raise ValueError("Gaussian scene_path is not bound to exactly one persistent integrity reference.")

    try:
        import torch
    except Exception as exc:  # pragma: no cover - production dependency guard
        raise RuntimeError("PyTorch is required to read the persisted Gaussian tensor artifact.") from exc

    try:
        payload: dict[str, Any] = torch.load(scene_path, map_location="cpu", weights_only=True)
    except TypeError as exc:  # fail closed; do not unsafe-unpickle a persisted artifact
        raise RuntimeError("This PyTorch build does not support safe weights_only Gaussian loading.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Persisted Gaussian scene payload must be a mapping.")
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported persisted Gaussian scene schema version.")
    if bool(payload.get("canonical_geometry_authoritative", False)):
        raise ValueError("Persisted Gaussian payload claims forbidden canonical geometry authority.")
    if bool(payload.get("certified_metrology", False)):
        raise ValueError("Persisted Gaussian payload claims forbidden certified metrology.")

    def array(name: str) -> np.ndarray:
        value = payload.get(name)
        if value is None or not hasattr(value, "detach"):
            raise ValueError(f"Persisted Gaussian payload is missing tensor {name}.")
        return value.detach().cpu().numpy()

    positions = np.asarray(array("means_reconstruction"), dtype=np.float32)
    scales = np.asarray(array("scales_reconstruction"), dtype=np.float32)
    colors = np.asarray(array("colors_rgb"), dtype=np.float32)
    opacities = np.asarray(array("opacities"), dtype=np.float32).reshape(-1)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError("Persisted Gaussian means must be finite (N, 3).")
    total = int(positions.shape[0])
    expected = int(snapshot.get("gaussian_count", total) or total)
    if total < 1 or expected != total:
        raise ValueError("Persisted Gaussian count does not match its manifest evidence.")
    if scales.shape != positions.shape or colors.shape != positions.shape or opacities.shape != (total,):
        raise ValueError("Persisted Gaussian tensor counts are inconsistent.")

    rgba = _rgba(colors, opacities)
    sizes = _sizes(scales)
    indices = _display_indices(total, max_gaussians)
    positions = np.ascontiguousarray(positions[indices])
    rgba = np.ascontiguousarray(rgba[indices])
    sizes = np.ascontiguousarray(sizes[indices])

    return InteractiveGaussianScene(
        scene_id=scene.scene_id,
        artifact_id=artifact.artifact_id,
        backend_id=str(snapshot.get("backend_id", "")),
        backend_version=str(snapshot.get("backend_version", "")),
        scene_path=str(scene_path),
        total_gaussian_count=total,
        display_gaussian_count=int(indices.shape[0]),
        positions_reconstruction=positions,
        colors_rgba=rgba,
        sizes_px=sizes,
        metric_geometry_status=str(snapshot.get("metric_geometry_status", "unknown")),
    )


__all__ = [
    "InteractiveGaussianScene",
    "MAX_INTERACTIVE_GAUSSIANS",
    "load_interactive_gaussian_scene",
]
