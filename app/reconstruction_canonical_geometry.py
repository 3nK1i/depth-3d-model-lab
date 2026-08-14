"""Persistent canonical-surface loading for the Reconstruction viewer.

Phase 9.3B consumes only the integrity-bound ``surface_obj_path`` that already
belongs to the persistent canonical ``surface_mesh`` artifact.  The resulting
arrays are presentation resources: they do not replace the canonical artifact,
create metric scale, certify metrology, or reconstruct a ``SurfaceMeshResult``
from incomplete persisted metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from app.reconstruction_visualization import reconstruction_points_to_viewer
from core.reconstruction.orchestration import ReconstructionArtifactKind
from core.reconstruction.persistent_scene import (
    PersistentReconstructionScene,
    PersistentSceneArtifactRecord,
    PersistentSceneFileReference,
)

CANONICAL_SURFACE_RESOURCE_ROLE = "surface_obj_path"
_DEFAULT_VERTEX_RGBA = np.asarray([0.72, 0.78, 0.84, 1.0], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class CanonicalViewerMesh:
    """Validated presentation arrays derived from one persistent canonical OBJ."""

    scene_id: str
    artifact_id: str
    provider_id: str
    source_path: str
    source_sha256: str
    vertices_reconstruction: np.ndarray
    vertices_viewer: np.ndarray
    faces: np.ndarray
    vertex_colors_rgba: np.ndarray
    bounds_min_viewer: np.ndarray
    bounds_max_viewer: np.ndarray
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        vertices_reconstruction = np.asarray(self.vertices_reconstruction, dtype=np.float32)
        vertices_viewer = np.asarray(self.vertices_viewer, dtype=np.float32)
        faces = np.asarray(self.faces, dtype=np.uint32)
        colors = np.asarray(self.vertex_colors_rgba, dtype=np.float32)
        lower = np.asarray(self.bounds_min_viewer, dtype=np.float32)
        upper = np.asarray(self.bounds_max_viewer, dtype=np.float32)

        if self.certified_metrology:
            raise ValueError("Canonical viewer mesh cannot claim certified metrology.")
        if vertices_reconstruction.ndim != 2 or vertices_reconstruction.shape[1:] != (3,):
            raise ValueError("vertices_reconstruction must have shape (N, 3).")
        if vertices_reconstruction.shape[0] < 3 or not np.isfinite(vertices_reconstruction).all():
            raise ValueError("Canonical viewer mesh requires at least three finite vertices.")
        if vertices_viewer.shape != vertices_reconstruction.shape or not np.isfinite(vertices_viewer).all():
            raise ValueError("vertices_viewer must be finite and match reconstruction vertices.")
        if faces.ndim != 2 or faces.shape[1:] != (3,) or faces.shape[0] < 1:
            raise ValueError("faces must contain at least one triangular face.")
        if np.any(faces >= vertices_reconstruction.shape[0]):
            raise ValueError("Canonical viewer faces reference vertices outside the mesh.")
        if colors.shape != (vertices_reconstruction.shape[0], 4):
            raise ValueError("vertex_colors_rgba must have shape (N, 4).")
        if not np.isfinite(colors).all() or np.any(colors < 0.0) or np.any(colors > 1.0):
            raise ValueError("vertex_colors_rgba must be finite in [0, 1].")
        if lower.shape != (3,) or upper.shape != (3,) or np.any(upper < lower):
            raise ValueError("Canonical viewer bounds must be ordered 3-vectors.")

        for name, value in (
            ("vertices_reconstruction", vertices_reconstruction),
            ("vertices_viewer", vertices_viewer),
            ("faces", faces),
            ("vertex_colors_rgba", colors),
            ("bounds_min_viewer", lower),
            ("bounds_max_viewer", upper),
        ):
            array = np.ascontiguousarray(value)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "certified_metrology", False)

    @property
    def vertex_count(self) -> int:
        return int(self.vertices_viewer.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_artifact(scene: PersistentReconstructionScene) -> PersistentSceneArtifactRecord:
    artifact = next(
        (item for item in scene.artifacts if item.artifact_id == scene.canonical_geometry_artifact_id),
        None,
    )
    if artifact is None:
        raise ValueError("Persistent scene canonical artifact is unavailable.")
    if not artifact.canonical_geometry_authoritative:
        raise ValueError("Persistent canonical artifact lost canonical geometry authority.")
    if artifact.kind is not ReconstructionArtifactKind.SURFACE_MESH:
        raise ValueError("Phase 9.3B requires the persistent canonical surface_mesh artifact.")
    return artifact


def canonical_surface_reference(
    scene: PersistentReconstructionScene,
) -> tuple[PersistentSceneArtifactRecord, PersistentSceneFileReference] | None:
    """Return the exact canonical ``surface_obj_path`` reference, if persisted."""

    if not isinstance(scene, PersistentReconstructionScene):
        raise TypeError("scene must be PersistentReconstructionScene.")
    artifact = _canonical_artifact(scene)
    reference = next(
        (
            item
            for item in scene.file_references
            if item.artifact_id == artifact.artifact_id
            and item.role == CANONICAL_SURFACE_RESOURCE_ROLE
        ),
        None,
    )
    if reference is None:
        return None
    return artifact, reference


def _verify_reference(reference: PersistentSceneFileReference) -> Path:
    path = Path(reference.path)
    if not path.is_file():
        raise OSError(f"Persistent canonical surface is missing: {path}")
    if path.stat().st_size != reference.size_bytes:
        raise ValueError(f"Persistent canonical surface size changed: {path}")
    digest = _sha256_file(path)
    if digest != reference.sha256:
        raise ValueError(f"Persistent canonical surface checksum changed: {path}")
    return path


def _parse_obj_vertex(tokens: list[str], *, line_number: int) -> tuple[np.ndarray, np.ndarray]:
    if len(tokens) < 4:
        raise ValueError(f"Canonical OBJ line {line_number} has an invalid vertex record.")
    try:
        xyz = np.asarray([float(tokens[1]), float(tokens[2]), float(tokens[3])], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"Canonical OBJ line {line_number} has a non-numeric vertex.") from exc
    if not np.isfinite(xyz).all():
        raise ValueError(f"Canonical OBJ line {line_number} has a non-finite vertex.")

    color = _DEFAULT_VERTEX_RGBA.copy()
    if len(tokens) >= 7:
        try:
            rgb = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f"Canonical OBJ line {line_number} has invalid vertex color values.") from exc
        if not np.isfinite(rgb).all():
            raise ValueError(f"Canonical OBJ line {line_number} has non-finite vertex color values.")
        # The project writer emits normalized RGB. Accept byte-range RGB as a
        # conservative compatibility convenience without changing geometry.
        if np.any(rgb > 1.0):
            if np.any(rgb > 255.0) or np.any(rgb < 0.0):
                raise ValueError(f"Canonical OBJ line {line_number} has RGB values outside [0, 255].")
            rgb = rgb / 255.0
        if np.any(rgb < 0.0) or np.any(rgb > 1.0):
            raise ValueError(f"Canonical OBJ line {line_number} has RGB values outside [0, 1].")
        color[:3] = rgb.astype(np.float32)
    return xyz, color


def _obj_vertex_index(token: str, *, vertex_count: int, line_number: int) -> int:
    raw = token.split("/", 1)[0].strip()
    if not raw:
        raise ValueError(f"Canonical OBJ line {line_number} has an empty face index.")
    try:
        index = int(raw)
    except ValueError as exc:
        raise ValueError(f"Canonical OBJ line {line_number} has an invalid face index.") from exc
    if index == 0:
        raise ValueError(f"Canonical OBJ line {line_number} uses forbidden OBJ index 0.")
    resolved = index - 1 if index > 0 else vertex_count + index
    if resolved < 0 or resolved >= vertex_count:
        raise ValueError(f"Canonical OBJ line {line_number} references a missing vertex.")
    return resolved


def _parse_canonical_surface_obj(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            record = tokens[0]
            if record == "v":
                xyz, color = _parse_obj_vertex(tokens, line_number=line_number)
                vertices.append(xyz)
                colors.append(color)
            elif record == "f":
                if len(tokens) != 4:
                    raise ValueError(
                        f"Canonical OBJ line {line_number} is not a triangular face; "
                        "Phase 9.3B consumes the project's triangular surface format only."
                    )
                if len(vertices) < 3:
                    raise ValueError(f"Canonical OBJ line {line_number} defines a face before enough vertices.")
                face = tuple(
                    _obj_vertex_index(token, vertex_count=len(vertices), line_number=line_number)
                    for token in tokens[1:]
                )
                if len(set(face)) != 3:
                    raise ValueError(f"Canonical OBJ line {line_number} contains a degenerate face.")
                faces.append(face)  # type: ignore[arg-type]
            elif record in {"vn", "vt", "o", "g", "s", "usemtl", "mtllib"}:
                # Persisted Phase 8 surfaces may include normals. VisPy computes
                # display shading from the loaded triangles, so these records are
                # presentation-optional and do not need to be promoted into a
                # reconstructed SurfaceMeshResult.
                continue
            else:
                raise ValueError(f"Canonical OBJ line {line_number} uses unsupported record {record!r}.")

    if len(vertices) < 3 or not faces:
        raise ValueError("Canonical OBJ does not contain a renderable triangular surface.")
    vertex_array = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.uint32)
    color_array = np.asarray(colors, dtype=np.float32)
    return vertex_array, face_array, color_array


def load_canonical_viewer_mesh(
    scene: PersistentReconstructionScene,
) -> CanonicalViewerMesh | None:
    """Load the integrity-bound persistent canonical OBJ for presentation only."""

    resolved = canonical_surface_reference(scene)
    if resolved is None:
        return None
    artifact, reference = resolved
    path = _verify_reference(reference)
    vertices_reconstruction, faces, colors = _parse_canonical_surface_obj(path)

    snapshot = dict(artifact.payload_snapshot)
    expected_vertices = snapshot.get("surface_vertex_count")
    expected_faces = snapshot.get("surface_face_count")
    if expected_vertices is not None and int(expected_vertices) != int(vertices_reconstruction.shape[0]):
        raise ValueError("Persistent canonical surface vertex count disagrees with artifact metadata.")
    if expected_faces is not None and int(expected_faces) != int(faces.shape[0]):
        raise ValueError("Persistent canonical surface face count disagrees with artifact metadata.")

    vertices_viewer = reconstruction_points_to_viewer(vertices_reconstruction).astype(np.float32)
    lower = vertices_viewer.min(axis=0)
    upper = vertices_viewer.max(axis=0)
    return CanonicalViewerMesh(
        scene_id=scene.scene_id,
        artifact_id=artifact.artifact_id,
        provider_id=artifact.provider_id,
        source_path=str(path.resolve()),
        source_sha256=reference.sha256,
        vertices_reconstruction=vertices_reconstruction.astype(np.float32),
        vertices_viewer=vertices_viewer,
        faces=faces,
        vertex_colors_rgba=colors,
        bounds_min_viewer=lower,
        bounds_max_viewer=upper,
        certified_metrology=False,
    )
