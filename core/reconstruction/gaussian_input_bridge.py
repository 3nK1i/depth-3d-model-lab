"""Phase 9.4B canonical-scene -> Gaussian input bridge.

The bridge materializes the existing backend-neutral :class:`NeuralSceneDataset`
from persistent canonical-scene evidence.  It never starts Gaussian training or
rendering and never promotes Gaussian appearance to geometry, metric,
measurement, QC, or metrology authority.

A source image is accepted only when its exact pixel fingerprint matches the
fingerprint explicitly persisted beside the canonical camera pose.  The bridge
therefore fails closed instead of pairing images and poses by filename or list
position alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from math import isfinite
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from core.contracts.validation import metadata_as_dict, require_nonempty
from core.reconstruction.da3 import read_da3_result_manifest, validate_da3_npz_result
from core.reconstruction.da3_provider import read_da3_input_manifest
from core.reconstruction.dense_fusion import DenseSceneProfile
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.gaussian_splatting import GaussianSceneRole
from core.reconstruction.neural_scene import (
    NEURAL_SCENE_POSE_CONVENTION,
    NeuralSceneDataset,
    NeuralSceneFrame,
    neural_scene_dataset_fingerprint,
    write_neural_scene_dataset_manifest,
)
from core.reconstruction.orchestration import (
    ReconstructionArtifactAuthority,
    ReconstructionArtifactKind,
)
from core.reconstruction.persistent_scene import (
    PersistentReconstructionScene,
    PersistentSceneArtifactRecord,
)
from core.reconstruction.scale_reference import MetricGeometryStatus
from core.reconstruction.sfm_contracts import SfmScaleStatus, SfmWorldAlignmentStatus

PHASE9_4B_GAUSSIAN_INPUT_BRIDGE_SCHEMA_VERSION = 1
PHASE9_4B_CAMERA_POSE_CONVENTION = "T_reconstruction_from_camera"


class GaussianInputBridgeState(str, Enum):
    READY = "ready"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class GaussianInputFrameEvidence:
    """Exact source image plus intrinsics used to materialize one Gaussian view."""

    source_image_path: str
    source_frame_sha256: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        path = Path(require_nonempty(self.source_image_path, field_name="source_image_path"))
        if not path.is_absolute():
            raise ReconstructionValidationError("Gaussian source_image_path must be absolute.")
        digest = str(self.source_frame_sha256).strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ReconstructionValidationError("source_frame_sha256 must be a SHA-256 hex digest.")
        width = int(self.width)
        height = int(self.height)
        if width < 2 or height < 2:
            raise ReconstructionValidationError("Gaussian input width/height must be >= 2.")
        fx = float(self.fx)
        fy = float(self.fy)
        cx = float(self.cx)
        cy = float(self.cy)
        if not all(isfinite(value) for value in (fx, fy, cx, cy)) or fx <= 0.0 or fy <= 0.0:
            raise ReconstructionValidationError("Gaussian input intrinsics must be finite with positive focal lengths.")
        if not 0.0 <= cx <= float(width) or not 0.0 <= cy <= float(height):
            raise ReconstructionValidationError("Gaussian input principal point must lie inside the image extent.")
        object.__setattr__(self, "source_image_path", str(path))
        object.__setattr__(self, "source_frame_sha256", digest)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "fx", fx)
        object.__setattr__(self, "fy", fy)
        object.__setattr__(self, "cx", cx)
        object.__setattr__(self, "cy", cy)


@dataclass(frozen=True, slots=True)
class _CanonicalCameraBundle:
    artifact_id: str
    provider_id: str
    reconstruction_frame_id: str
    frame_ids: tuple[str, ...]
    pose_layers: tuple[str, ...]
    source_frame_sha256: tuple[str, ...]
    matrices: np.ndarray

    def __post_init__(self) -> None:
        matrices = np.asarray(self.matrices, dtype=np.float64)
        count = matrices.shape[0] if matrices.ndim == 3 else 0
        if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or count < 1:
            raise ReconstructionValidationError("Canonical camera matrices must have shape (N,4,4).")
        if not np.isfinite(matrices).all():
            raise ReconstructionValidationError("Canonical camera matrices must be finite.")
        if len(self.frame_ids) != count or len(self.pose_layers) != count or len(self.source_frame_sha256) != count:
            raise ReconstructionValidationError(
                "Gaussian input bridge requires one frame ID, pose layer, and source fingerprint per canonical camera."
            )
        if len(set(self.frame_ids)) != count or len(set(self.source_frame_sha256)) != count:
            raise ReconstructionValidationError("Canonical camera frame IDs and source fingerprints must be unique.")
        for digest in self.source_frame_sha256:
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ReconstructionValidationError("Canonical camera source fingerprints must be SHA-256 digests.")
        for index, matrix in enumerate(matrices):
            if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
                raise ReconstructionValidationError(f"Canonical camera pose {index} is not homogeneous.")
            rotation = matrix[:3, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
                np.linalg.det(rotation), 1.0, atol=1e-5
            ):
                raise ReconstructionValidationError(f"Canonical camera pose {index} is not rigid.")
        frozen = np.ascontiguousarray(matrices)
        frozen.setflags(write=False)
        object.__setattr__(self, "matrices", frozen)


@dataclass(frozen=True, slots=True)
class GaussianCanonicalInputBridgeResult:
    """Ready Gaussian input bundle bound to one persistent canonical scene."""

    scene_id: str
    camera_artifact_id: str
    dataset: NeuralSceneDataset
    manifest_path: str
    dataset_fingerprint_sha256: str
    source_frame_sha256: tuple[str, ...]
    source_kind: str
    scene_role: GaussianSceneRole = GaussianSceneRole.APPEARANCE_ONLY
    canonical_geometry_authoritative: bool = False
    metric_scale_authoritative: bool = False
    measurement_authoritative: bool = False
    qc_authoritative: bool = False
    certified_metrology: bool = False
    training_started: bool = False
    rendering_started: bool = False
    schema_version: int = PHASE9_4B_GAUSSIAN_INPUT_BRIDGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, NeuralSceneDataset):
            raise TypeError("dataset must be NeuralSceneDataset.")
        object.__setattr__(self, "scene_id", require_nonempty(self.scene_id, field_name="scene_id"))
        object.__setattr__(
            self, "camera_artifact_id", require_nonempty(self.camera_artifact_id, field_name="camera_artifact_id")
        )
        object.__setattr__(self, "source_kind", require_nonempty(self.source_kind, field_name="source_kind"))
        hashes = tuple(str(value).strip().lower() for value in self.source_frame_sha256)
        if len(hashes) != self.dataset.frame_count or len(set(hashes)) != len(hashes):
            raise ReconstructionValidationError("Gaussian input result requires one unique source fingerprint per frame.")
        for digest in hashes:
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ReconstructionValidationError("Gaussian input result source fingerprints must be SHA-256 digests.")
        object.__setattr__(self, "source_frame_sha256", hashes)
        manifest = Path(self.manifest_path)
        if not manifest.is_absolute():
            raise ReconstructionValidationError("Gaussian input manifest_path must be absolute.")
        expected = neural_scene_dataset_fingerprint(self.dataset)
        digest = str(self.dataset_fingerprint_sha256).strip().lower()
        if digest != expected:
            raise ReconstructionValidationError("Gaussian input dataset fingerprint mismatch.")
        if self.scene_role is not GaussianSceneRole.APPEARANCE_ONLY:
            raise ReconstructionValidationError("Gaussian canonical input must remain appearance-only.")
        if any(
            bool(value)
            for value in (
                self.canonical_geometry_authoritative,
                self.metric_scale_authoritative,
                self.measurement_authoritative,
                self.qc_authoritative,
                self.certified_metrology,
                self.training_started,
                self.rendering_started,
            )
        ):
            raise ReconstructionValidationError(
                "Gaussian input preparation cannot claim authority or start training/rendering."
            )
        object.__setattr__(self, "manifest_path", str(manifest))
        object.__setattr__(self, "dataset_fingerprint_sha256", expected)
        object.__setattr__(self, "scene_role", GaussianSceneRole.APPEARANCE_ONLY)
        object.__setattr__(self, "canonical_geometry_authoritative", False)
        object.__setattr__(self, "metric_scale_authoritative", False)
        object.__setattr__(self, "measurement_authoritative", False)
        object.__setattr__(self, "qc_authoritative", False)
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "training_started", False)
        object.__setattr__(self, "rendering_started", False)
        object.__setattr__(self, "schema_version", PHASE9_4B_GAUSSIAN_INPUT_BRIDGE_SCHEMA_VERSION)

    @property
    def frame_count(self) -> int:
        return self.dataset.frame_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "camera_artifact_id": self.camera_artifact_id,
            "source_kind": self.source_kind,
            "manifest_path": self.manifest_path,
            "dataset_fingerprint_sha256": self.dataset_fingerprint_sha256,
            "frame_count": self.frame_count,
            "source_frame_sha256": list(self.source_frame_sha256),
            "reconstruction_frame_id": self.dataset.reconstruction_frame_id,
            "metric_geometry_status": self.dataset.metric_geometry_status.value,
            "unit_to_metre": self.dataset.unit_to_metre,
            "scene_role": self.scene_role.value,
            "canonical_geometry_authoritative": False,
            "metric_scale_authoritative": False,
            "measurement_authoritative": False,
            "qc_authoritative": False,
            "certified_metrology": False,
            "training_started": False,
            "rendering_started": False,
        }


def _artifact(scene: PersistentReconstructionScene, kind: ReconstructionArtifactKind) -> PersistentSceneArtifactRecord:
    matches = tuple(item for item in scene.artifacts if item.kind is kind)
    if len(matches) != 1:
        raise ReconstructionValidationError(
            f"Persistent scene must contain exactly one {kind.value} artifact for Gaussian input preparation."
        )
    return matches[0]


def _canonical_camera_bundle(scene: PersistentReconstructionScene) -> _CanonicalCameraBundle:
    artifact = _artifact(scene, ReconstructionArtifactKind.CAMERA_TRAJECTORY)
    if artifact.authority is not ReconstructionArtifactAuthority.SUPPORTING_GEOMETRY:
        raise ReconstructionValidationError("Canonical camera trajectory must remain supporting geometry evidence.")
    if artifact.canonical_geometry_authoritative:
        raise ReconstructionValidationError("Canonical camera trajectory cannot claim canonical geometry authority.")
    snapshot = metadata_as_dict(artifact.payload_snapshot)  # type: ignore[arg-type]
    if bool(snapshot.get("certified_metrology", False)):
        raise ReconstructionValidationError("Canonical camera snapshot cannot certify metrology.")
    if str(snapshot.get("pose_convention", "")).strip() != PHASE9_4B_CAMERA_POSE_CONVENTION:
        raise ReconstructionValidationError("Canonical camera pose convention is unavailable or unsupported.")
    raw_matrices = snapshot.get("T_reconstruction_from_camera")
    raw_ids = snapshot.get("frame_ids")
    raw_layers = snapshot.get("pose_layers")
    raw_hashes = snapshot.get("source_frame_sha256")
    if not isinstance(raw_matrices, list) or not isinstance(raw_ids, list):
        raise ReconstructionValidationError("Persistent canonical scene does not contain explicit camera poses.")
    if not isinstance(raw_layers, list) or not isinstance(raw_hashes, list) or not raw_hashes:
        raise ReconstructionValidationError(
            "Persistent canonical cameras require explicit pose layers and source-frame fingerprints for Gaussian binding."
        )
    reconstruction_frame_id = require_nonempty(
        snapshot.get("reconstruction_frame_id", ""), field_name="reconstruction_frame_id"
    )
    return _CanonicalCameraBundle(
        artifact_id=artifact.artifact_id,
        provider_id=artifact.provider_id,
        reconstruction_frame_id=reconstruction_frame_id,
        frame_ids=tuple(str(value).strip() for value in raw_ids),
        pose_layers=tuple(str(value).strip() for value in raw_layers),
        source_frame_sha256=tuple(str(value).strip().lower() for value in raw_hashes),
        matrices=np.asarray(raw_matrices, dtype=np.float64),
    )


def _scene_scale(scene: PersistentReconstructionScene) -> tuple[MetricGeometryStatus, SfmScaleStatus, str, float | None, float | None, SfmWorldAlignmentStatus]:
    surface = _artifact(scene, ReconstructionArtifactKind.SURFACE_MESH)
    if not surface.canonical_geometry_authoritative:
        raise ReconstructionValidationError("Persistent surface mesh must remain canonical geometry authority.")
    snapshot = metadata_as_dict(surface.payload_snapshot)  # type: ignore[arg-type]
    raw_status = str(snapshot.get("metric_geometry_status", "")).strip()
    if not raw_status:
        raise ReconstructionValidationError("Persistent canonical surface lacks metric_geometry_status.")
    try:
        metric = MetricGeometryStatus(raw_status)
    except ValueError as exc:
        raise ReconstructionValidationError(
            f"Unsupported canonical metric_geometry_status for Gaussian input: {raw_status!r}."
        ) from exc
    unit_raw = snapshot.get("unit_to_metre")
    unit = None if unit_raw is None else float(unit_raw)
    if metric is MetricGeometryStatus.ARBITRARY:
        if unit is not None:
            raise ReconstructionValidationError("Arbitrary canonical geometry must not expose unit_to_metre.")
        scale_status = SfmScaleStatus.ARBITRARY
        relative = None
    else:
        if unit is None or not isfinite(unit) or unit <= 0.0:
            raise ReconstructionValidationError("Metric canonical geometry requires positive unit_to_metre.")
        scale_status = (
            SfmScaleStatus.CALIBRATED_METRIC
            if metric is MetricGeometryStatus.CALIBRATED_METRIC
            else SfmScaleStatus.ESTIMATED_METRIC
        )
        relative_raw = snapshot.get("relative_scale_uncertainty")
        relative = None if relative_raw is None else float(relative_raw)
        if relative is not None and (not isfinite(relative) or relative < 0.0):
            raise ReconstructionValidationError("relative_scale_uncertainty must be finite and non-negative.")
    alignment_raw = str(snapshot.get("world_alignment_status", SfmWorldAlignmentStatus.ARBITRARY.value)).strip()
    try:
        alignment = SfmWorldAlignmentStatus(alignment_raw)
    except ValueError as exc:
        raise ReconstructionValidationError("Unsupported canonical world_alignment_status.") from exc
    scale_profile_id = str(snapshot.get("scale_profile_id", "")).strip()
    if not scale_profile_id:
        dense = _artifact(scene, ReconstructionArtifactKind.DENSE_GEOMETRY)
        dense_snapshot = metadata_as_dict(dense.payload_snapshot)  # type: ignore[arg-type]
        scale_profile_id = str(dense_snapshot.get("scale_profile_id", "")).strip()
    if not scale_profile_id:
        scale_profile_id = f"phase9-4b:{scene.scene_id}:preserved-scale-status"
    return metric, scale_status, scale_profile_id, unit, relative, alignment


def _safe_png_name(index: int, frame_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(frame_id)).strip("-.") or f"frame-{index:04d}"
    return f"{index:04d}_{safe}.png"


def _source_image_sha256(image_bgr: np.ndarray) -> str:
    image = np.ascontiguousarray(image_bgr, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ReconstructionValidationError("Gaussian source must decode to HxWx3 BGR uint8.")
    digest = sha256()
    digest.update(str(tuple(int(value) for value in image.shape)).encode("ascii"))
    digest.update(b"\0uint8\0")
    digest.update(image.tobytes(order="C"))
    return digest.hexdigest()


def _decode_and_verify_source(evidence: GaussianInputFrameEvidence) -> np.ndarray:
    path = Path(evidence.source_image_path)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ReconstructionValidationError(f"Unable to decode Gaussian source image: {path}")
    digest = _source_image_sha256(bgr)
    if digest != evidence.source_frame_sha256:
        raise ReconstructionValidationError(
            f"Gaussian source image fingerprint mismatch for {path.name}; refusing image/pose pairing."
        )
    return bgr


def materialize_gaussian_input_from_canonical_scene(
    scene: PersistentReconstructionScene,
    source_frames: Sequence[GaussianInputFrameEvidence],
    *,
    output_dir: str | Path,
    scene_profile: DenseSceneProfile = DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
    source_kind: str = "explicit_source_evidence",
) -> GaussianCanonicalInputBridgeResult:
    """Materialize a backend-neutral Gaussian dataset from exact canonical evidence."""

    if not isinstance(scene, PersistentReconstructionScene):
        raise TypeError("scene must be PersistentReconstructionScene.")
    if not scene.canonical_geometry_authoritative:
        raise ReconstructionValidationError("Gaussian input requires an authoritative canonical scene.")
    if not isinstance(scene_profile, DenseSceneProfile):
        raise TypeError("scene_profile must be DenseSceneProfile.")
    camera = _canonical_camera_bundle(scene)
    evidence = tuple(source_frames)
    if not evidence or any(not isinstance(item, GaussianInputFrameEvidence) for item in evidence):
        raise ReconstructionValidationError("Gaussian input requires explicit source-frame evidence.")
    by_digest = {item.source_frame_sha256: item for item in evidence}
    if len(by_digest) != len(evidence):
        raise ReconstructionValidationError("Gaussian source-frame fingerprints must be unique.")
    if set(by_digest) != set(camera.source_frame_sha256):
        raise ReconstructionValidationError(
            "Gaussian source evidence must exactly cover the persisted canonical camera fingerprints."
        )

    metric, scale_status, scale_profile_id, unit, relative, alignment = _scene_scale(scene)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    frames: list[NeuralSceneFrame] = []
    for index, (frame_id, pose_layer, digest, matrix) in enumerate(
        zip(camera.frame_ids, camera.pose_layers, camera.source_frame_sha256, camera.matrices)
    ):
        item = by_digest[digest]
        bgr = _decode_and_verify_source(item)
        if (bgr.shape[1], bgr.shape[0]) != (item.width, item.height):
            interpolation = cv2.INTER_AREA if bgr.shape[1] >= item.width and bgr.shape[0] >= item.height else cv2.INTER_LINEAR
            bgr = cv2.resize(bgr, (item.width, item.height), interpolation=interpolation)
        target = (image_dir / _safe_png_name(index, frame_id)).resolve()
        if not cv2.imwrite(str(target), bgr):
            raise ReconstructionValidationError(f"Unable to materialize Gaussian input image: {target}")
        frames.append(
            NeuralSceneFrame(
                frame_id=frame_id,
                image_path=str(target),
                width=item.width,
                height=item.height,
                fx=item.fx,
                fy=item.fy,
                cx=item.cx,
                cy=item.cy,
                T_reconstruction_from_camera=matrix,
                pose_layer=pose_layer,
                camera_profile_id=f"phase9-4b:{camera.artifact_id}",
            )
        )

    dataset = NeuralSceneDataset(
        session_id=scene.session_id,
        scene_profile=scene_profile,
        reconstruction_frame_id=camera.reconstruction_frame_id,
        camera_profile_id=f"phase9-4b:{camera.artifact_id}",
        scale_profile_id=scale_profile_id,
        metric_geometry_status=metric,
        scale_status=scale_status,
        world_alignment_status=alignment,
        unit_to_metre=unit,
        relative_scale_uncertainty=relative,
        frames=tuple(frames),
        certified_metrology=False,
        pose_convention=NEURAL_SCENE_POSE_CONVENTION,
    )
    manifest = write_neural_scene_dataset_manifest(dataset, (root / "dataset.json").resolve())
    return GaussianCanonicalInputBridgeResult(
        scene_id=scene.scene_id,
        camera_artifact_id=camera.artifact_id,
        dataset=dataset,
        manifest_path=str(manifest),
        dataset_fingerprint_sha256=neural_scene_dataset_fingerprint(dataset),
        source_frame_sha256=camera.source_frame_sha256,
        source_kind=require_nonempty(source_kind, field_name="source_kind"),
    )


def _da3_sidecar_paths(scene: PersistentReconstructionScene) -> tuple[Path, Path]:
    depth = _artifact(scene, ReconstructionArtifactKind.DEPTH_EVIDENCE)
    snapshot = metadata_as_dict(depth.payload_snapshot)  # type: ignore[arg-type]
    input_text = str(snapshot.get("da3_input_manifest_path", "")).strip()
    result_text = str(snapshot.get("da3_result_manifest_path", "")).strip()
    if not input_text or not result_text:
        raise ReconstructionValidationError(
            "Persistent scene does not retain the DA3 input/result sidecars required for exact Gaussian source binding."
        )
    input_path = Path(input_text)
    result_path = Path(result_text)
    if not input_path.is_absolute() or not result_path.is_absolute():
        raise ReconstructionValidationError("Persisted DA3 sidecar paths must be absolute.")
    if not input_path.is_file() or not result_path.is_file():
        raise ReconstructionValidationError("Persisted DA3 sidecar required for Gaussian input is unavailable.")
    return input_path, result_path


def materialize_gaussian_input_from_da3_persistent_scene(
    scene: PersistentReconstructionScene,
    *,
    output_dir: str | Path,
) -> GaussianCanonicalInputBridgeResult:
    """Materialize exact Gaussian inputs from a Phase 9.3H/J DA3 canonical scene.

    This adapter is read-only with respect to the persistent scene and DA3
    sidecars.  It reuses already-computed DA3 intrinsics and canonical poses; it
    never reruns DA3 and never starts a Gaussian backend.
    """

    camera = _canonical_camera_bundle(scene)
    input_path, result_path = _da3_sidecar_paths(scene)
    input_manifest = read_da3_input_manifest(input_path)
    result_manifest = read_da3_result_manifest(result_path)
    if input_manifest.request_id != result_manifest.request_id:
        raise ReconstructionValidationError("Persisted DA3 input/result request IDs differ.")
    arrays = validate_da3_npz_result(result_manifest)
    count = result_manifest.frame_count
    if len(input_manifest.image_paths) != count or len(camera.frame_ids) != count:
        raise ReconstructionValidationError("DA3 source/canonical camera frame counts do not match.")
    if not np.allclose(
        np.asarray(arrays["T_reconstruction_from_camera"], dtype=np.float64),
        camera.matrices,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ReconstructionValidationError(
            "Persisted canonical camera poses drift from the retained DA3 pose evidence."
        )
    intrinsics = np.asarray(arrays["intrinsics"], dtype=np.float64)
    source_frames: list[GaussianInputFrameEvidence] = []
    for index, path_text in enumerate(input_manifest.image_paths):
        path = Path(path_text)
        if not path.is_file():
            raise ReconstructionValidationError(f"Persisted DA3 source image is unavailable: {path}")
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ReconstructionValidationError(f"Unable to decode persisted DA3 source image: {path}")
        digest = _source_image_sha256(bgr)
        expected = camera.source_frame_sha256[index]
        if digest != expected:
            raise ReconstructionValidationError(
                "Persisted DA3 source-frame fingerprint no longer matches the canonical camera binding."
            )
        K = intrinsics[index]
        source_frames.append(
            GaussianInputFrameEvidence(
                source_image_path=str(path.resolve()),
                source_frame_sha256=digest,
                width=result_manifest.width,
                height=result_manifest.height,
                fx=float(K[0, 0]),
                fy=float(K[1, 1]),
                cx=float(K[0, 2]),
                cy=float(K[1, 2]),
            )
        )
    return materialize_gaussian_input_from_canonical_scene(
        scene,
        source_frames,
        output_dir=output_dir,
        scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
        source_kind="persisted_da3_sidecars",
    )


__all__ = [
    "PHASE9_4B_GAUSSIAN_INPUT_BRIDGE_SCHEMA_VERSION",
    "GaussianInputBridgeState",
    "GaussianInputFrameEvidence",
    "GaussianCanonicalInputBridgeResult",
    "materialize_gaussian_input_from_canonical_scene",
    "materialize_gaussian_input_from_da3_persistent_scene",
]
