"""Authority-preserving camera bridge for the I4.1.9A House MVS control.

The bridge consumes the already-qualified I4.1.7 camera solution.  It does not
estimate cameras, run SfM, run bundle adjustment, undistort images, start
PatchMatch Stereo, or start Stereo Fusion.  Its only geometric operation is to
serialize the existing 40 authoritative SfM cameras together with the 28
already-localized extra observations into a derived COLMAP sparse model that
can be consumed by the later MVS stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.poses import rotation_matrix_to_quaternion_wxyz
from core.reconstruction.sfm_coordinates import (
    internal_pose_to_colmap_camera_from_reconstruction,
    validate_rigid_matrix,
)

MVS_CAMERA_BRIDGE_SCHEMA_VERSION = 1
MVS_CAMERA_BRIDGE_EXPECTED_GLOBAL_VIEWS = 40
MVS_CAMERA_BRIDGE_EXPECTED_LOCALIZED_VIEWS = 28
MVS_CAMERA_BRIDGE_EXPECTED_TOTAL_VIEWS = 68

_LANDMARK_ID_RE = re.compile(r"^pycolmap-landmark-(\d+)$")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Could not read JSON evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"JSON evidence must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_clean_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ReconstructionValidationError(
            f"A1 output directory must be empty or absent; refusing to overwrite evidence: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def _safe_workspace_path(root: Path, relative: str, *, field_name: str) -> Path:
    rel = Path(str(relative))
    if rel.is_absolute() or ".." in rel.parts:
        raise ReconstructionValidationError(f"{field_name} must be workspace-relative: {relative}")
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ReconstructionValidationError(f"{field_name} escapes workspace: {relative}") from exc
    return candidate


def _float_list(value: object, *, field_name: str, length: int | None = None) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReconstructionValidationError(f"{field_name} must be an array.")
    result = [float(item) for item in value]
    if length is not None and len(result) != length:
        raise ReconstructionValidationError(f"{field_name} must contain {length} values.")
    if not all(isfinite(item) for item in result):
        raise ReconstructionValidationError(f"{field_name} contains non-finite values.")
    return result


def _pose(value: object, *, field_name: str) -> np.ndarray:
    return validate_rigid_matrix(value, field_name=field_name)


def _point3d_id(landmark_id: str) -> int:
    match = _LANDMARK_ID_RE.fullmatch(str(landmark_id))
    if match is None:
        raise ReconstructionValidationError(
            f"A1 only accepts authoritative PyCOLMAP landmark identifiers; got {landmark_id!r}."
        )
    return int(match.group(1))


@dataclass(frozen=True, slots=True)
class CameraBridgeView:
    source_sequence_index: int
    provenance: str
    image_name: str
    image_path: str
    camera_id: int
    T_reconstruction_from_camera: np.ndarray
    backend_image_id: int | None = None
    matched_point3d_ids: tuple[int, ...] = ()
    matched_feature_xy: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        index = int(self.source_sequence_index)
        if index < 0:
            raise ReconstructionValidationError("source_sequence_index must be non-negative.")
        object.__setattr__(self, "source_sequence_index", index)
        provenance = str(self.provenance).strip()
        if provenance not in {"global_sfm_anchor", "localized_extra_observation"}:
            raise ReconstructionValidationError(f"Unsupported A1 pose provenance: {provenance}")
        object.__setattr__(self, "provenance", provenance)
        name = str(self.image_name).strip()
        if not name or Path(name).name != name:
            raise ReconstructionValidationError("A1 image names must be non-empty basenames.")
        object.__setattr__(self, "image_name", name)
        object.__setattr__(self, "image_path", str(self.image_path))
        camera_id = int(self.camera_id)
        if camera_id < 0:
            raise ReconstructionValidationError("camera_id must be non-negative.")
        object.__setattr__(self, "camera_id", camera_id)
        if self.backend_image_id is not None:
            backend = int(self.backend_image_id)
            if backend < 0:
                raise ReconstructionValidationError("backend_image_id must be non-negative.")
            object.__setattr__(self, "backend_image_id", backend)
        object.__setattr__(
            self,
            "T_reconstruction_from_camera",
            _pose(self.T_reconstruction_from_camera, field_name="T_reconstruction_from_camera"),
        )
        point_ids = tuple(int(value) for value in self.matched_point3d_ids)
        xy = tuple((float(item[0]), float(item[1])) for item in self.matched_feature_xy)
        if len(point_ids) != len(xy):
            raise ReconstructionValidationError("Matched point IDs and feature coordinates must align.")
        if len(set(point_ids)) != len(point_ids):
            raise ReconstructionValidationError("One localized image cannot bind the same 3D point twice.")
        if not all(isfinite(coord) for pair in xy for coord in pair):
            raise ReconstructionValidationError("Localized feature coordinates must be finite.")
        object.__setattr__(self, "matched_point3d_ids", point_ids)
        object.__setattr__(self, "matched_feature_xy", xy)
        if provenance == "global_sfm_anchor" and point_ids:
            raise ReconstructionValidationError("Global views inherit native tracks and must not carry A1 match payloads.")
        if provenance == "localized_extra_observation" and not point_ids:
            raise ReconstructionValidationError("Localized extra views require accepted sparse landmark observations.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_sequence_index": self.source_sequence_index,
            "provenance": self.provenance,
            "image_name": self.image_name,
            "image_path": self.image_path,
            "camera_id": self.camera_id,
            "backend_image_id": self.backend_image_id,
            "T_reconstruction_from_camera": self.T_reconstruction_from_camera.tolist(),
            "matched_point3d_ids": list(self.matched_point3d_ids),
            "matched_feature_xy": [list(value) for value in self.matched_feature_xy],
        }


@dataclass(frozen=True, slots=True)
class CameraBridgeInputs:
    i4_1_7_dir: Path
    i4_1_7_summary: Path
    orbit_diagnostics: Path
    sfm_report: Path
    native_sparse_model: Path
    localization_dir: Path
    selected_model_id: int
    shared_camera_id: int
    views: tuple[CameraBridgeView, ...]
    camera_model_payload: Mapping[str, Any]
    native_workspace_root: Path

    @property
    def global_views(self) -> tuple[CameraBridgeView, ...]:
        return tuple(item for item in self.views if item.provenance == "global_sfm_anchor")

    @property
    def localized_views(self) -> tuple[CameraBridgeView, ...]:
        return tuple(item for item in self.views if item.provenance == "localized_extra_observation")


def _selected_model_id(sfm_payload: Mapping[str, Any]) -> int:
    metadata = sfm_payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ReconstructionValidationError("Authoritative SfM payload metadata is missing.")
    selected = metadata.get("selected_model_id")
    if selected is None:
        raise ReconstructionValidationError("Authoritative SfM payload lacks selected_model_id.")
    value = int(selected)
    if value < 0:
        raise ReconstructionValidationError("selected_model_id must be non-negative.")
    return value


def load_house_camera_bridge_inputs(i4_1_7_dir: str | Path) -> CameraBridgeInputs:
    """Load and cross-check the exact I4.1.7 House camera evidence."""

    root = Path(i4_1_7_dir).expanduser().resolve()
    if not root.is_dir():
        raise ReconstructionValidationError(
            f"I4.1.7 House workspace is missing; A1 fails closed instead of re-running SfM: {root}"
        )

    summary_path = root / "i4_1_7_summary.json"
    diagnostics_path = root / "i4_1_7_orbit_diagnostics.npz"
    sfm_report_path = root / "01_i1_baseline/01_sfm/real_sfm_report.json"
    localization_dir = root / "02_decoupled_localization"
    for path, description in (
        (summary_path, "I4.1.7 summary"),
        (diagnostics_path, "I4.1.7 orbit diagnostics"),
        (sfm_report_path, "authoritative SfM report"),
        (localization_dir, "I4.1.7 localization directory"),
    ):
        if not path.exists():
            raise ReconstructionValidationError(f"Missing {description}; A1 fails closed: {path}")

    summary = _read_json(summary_path)
    if summary.get("phase") != "I4.1.7" or summary.get("status") != "qualified_for_i4_1_8":
        raise ReconstructionValidationError("A1 requires the qualified I4.1.7 House positive-control result.")
    authority = summary.get("authority", {})
    if not isinstance(authority, Mapping) or not bool(authority.get("single_authoritative_sfm_frame")):
        raise ReconstructionValidationError("I4.1.7 does not assert one authoritative SfM frame.")
    if bool(authority.get("metric_scale_claimed")) or bool(authority.get("measurement_authoritative")):
        raise ReconstructionValidationError("A1 refuses promoted metric/measurement authority.")

    assessment = summary.get("assessment", {})
    if not isinstance(assessment, Mapping):
        raise ReconstructionValidationError("I4.1.7 assessment is missing.")
    selected_sources = [int(v) for v in assessment.get("selected_source_sequence_indices", [])]
    selected_provenance = [str(v) for v in assessment.get("selected_pose_provenance", [])]
    if len(selected_sources) != len(selected_provenance):
        raise ReconstructionValidationError("I4.1.7 selected source/provenance arrays do not align.")
    if selected_sources != list(range(MVS_CAMERA_BRIDGE_EXPECTED_TOTAL_VIEWS)):
        raise ReconstructionValidationError(
            "House A1 requires all 68 source views (0..67); refusing a hidden/global view cap."
        )
    if selected_provenance.count("global_sfm_anchor") != MVS_CAMERA_BRIDGE_EXPECTED_GLOBAL_VIEWS:
        raise ReconstructionValidationError("House A1 expected exactly 40 global SfM anchors.")
    if selected_provenance.count("localized_extra_observation") != MVS_CAMERA_BRIDGE_EXPECTED_LOCALIZED_VIEWS:
        raise ReconstructionValidationError("House A1 expected exactly 28 localized extra observations.")

    try:
        diagnostics = np.load(diagnostics_path, allow_pickle=False)
        diag_sources = diagnostics["selected_source_sequence_indices"].astype(np.int64).tolist()
        diag_provenance = diagnostics["selected_pose_provenance"].astype(str).tolist()
        diag_matrices = np.asarray(diagnostics["selected_camera_matrices"], dtype=np.float64)
    except Exception as exc:
        raise ReconstructionValidationError(f"Could not read I4.1.7 orbit diagnostics: {exc}") from exc
    if diag_sources != selected_sources or diag_provenance != selected_provenance:
        raise ReconstructionValidationError("I4.1.7 summary and orbit diagnostics disagree on the selected 68 views.")
    if diag_matrices.shape != (MVS_CAMERA_BRIDGE_EXPECTED_TOTAL_VIEWS, 4, 4):
        raise ReconstructionValidationError("I4.1.7 orbit diagnostics do not contain 68 camera poses.")

    sfm_report = _read_json(sfm_report_path)
    sfm_payload = sfm_report.get("sfm_result")
    if not isinstance(sfm_payload, Mapping):
        raise ReconstructionValidationError("Authoritative SfM report lacks sfm_result.")
    gauge = sfm_payload.get("gauge", {})
    if not isinstance(gauge, Mapping) or gauge.get("unit_to_metre") is not None:
        raise ReconstructionValidationError("A1 only accepts the arbitrary-gauge authoritative SfM solution.")
    camera_models = sfm_payload.get("camera_models", [])
    if not isinstance(camera_models, Sequence) or isinstance(camera_models, (str, bytes)) or len(camera_models) != 1:
        raise ReconstructionValidationError("House A1 requires the existing one-shared-camera SfM solution.")
    camera_model = camera_models[0]
    if not isinstance(camera_model, Mapping):
        raise ReconstructionValidationError("SfM camera model payload is invalid.")
    shared_camera_id = int(camera_model.get("backend_camera_id", -1))
    if shared_camera_id < 0:
        raise ReconstructionValidationError("SfM camera model lacks a valid backend_camera_id.")

    workspace = sfm_payload.get("workspace")
    if not isinstance(workspace, Mapping):
        raise ReconstructionValidationError("Authoritative SfM workspace metadata is missing.")
    workspace_root = Path(str(workspace.get("workspace_root", ""))).expanduser().resolve()
    paths = workspace.get("paths", {})
    if not isinstance(paths, Mapping):
        raise ReconstructionValidationError("Authoritative SfM workspace paths are missing.")
    sparse_relative = str(paths.get("sparse_directory", ""))
    selected_model_id = _selected_model_id(sfm_payload)
    native_sparse_model = _safe_workspace_path(
        workspace_root,
        str(Path(sparse_relative) / str(selected_model_id)),
        field_name="native_sparse_model",
    )
    if not native_sparse_model.is_dir():
        raise ReconstructionValidationError(
            "The exact native sparse model no longer exists; A1 will not silently reconstruct it: "
            f"{native_sparse_model}"
        )

    workspace_images = workspace.get("images", [])
    cameras = sfm_payload.get("cameras", [])
    if not isinstance(workspace_images, Sequence) or isinstance(workspace_images, (str, bytes)):
        raise ReconstructionValidationError("SfM workspace image manifest is invalid.")
    if not isinstance(cameras, Sequence) or isinstance(cameras, (str, bytes)):
        raise ReconstructionValidationError("SfM camera estimates are invalid.")
    if len(cameras) != MVS_CAMERA_BRIDGE_EXPECTED_GLOBAL_VIEWS:
        raise ReconstructionValidationError("House A1 expected exactly 40 registered SfM cameras.")

    camera_by_keyframe: dict[str, Mapping[str, Any]] = {}
    for raw in cameras:
        if not isinstance(raw, Mapping):
            raise ReconstructionValidationError("SfM camera estimate must be an object.")
        keyframe_id = str(raw.get("keyframe_id", ""))
        if not keyframe_id or keyframe_id in camera_by_keyframe:
            raise ReconstructionValidationError("SfM camera keyframe IDs must be unique and non-empty.")
        camera_by_keyframe[keyframe_id] = raw

    global_by_source: dict[int, CameraBridgeView] = {}
    for raw in workspace_images:
        if not isinstance(raw, Mapping):
            raise ReconstructionValidationError("SfM workspace image must be an object.")
        keyframe_id = str(raw.get("keyframe_id", ""))
        camera = camera_by_keyframe.get(keyframe_id)
        if camera is None:
            continue
        source_index = int(raw.get("source_sequence_index", -1))
        staged_relative = str(raw.get("staged_image_path", ""))
        staged = _safe_workspace_path(workspace_root, staged_relative, field_name="staged_image_path")
        if not staged.is_file():
            raise ReconstructionValidationError(f"Authoritative staged SfM image is missing: {staged}")
        expected_digest = str(raw.get("staged_sha256", "")).strip().lower()
        actual_digest = sha256_file(staged)
        if expected_digest and actual_digest != expected_digest:
            raise ReconstructionValidationError(f"SfM staged image checksum changed: {staged}")
        pose = _pose(camera.get("T_reconstruction_from_camera"), field_name=f"global_pose[{source_index}]")
        backend_image_id = camera.get("backend_image_id")
        if backend_image_id is None:
            raise ReconstructionValidationError("A1 requires native backend image IDs for the 40 SfM anchors.")
        global_by_source[source_index] = CameraBridgeView(
            source_sequence_index=source_index,
            provenance="global_sfm_anchor",
            image_name=staged.name,
            image_path=str(staged),
            camera_id=shared_camera_id,
            backend_image_id=int(backend_image_id),
            T_reconstruction_from_camera=pose,
        )

    if len(global_by_source) != MVS_CAMERA_BRIDGE_EXPECTED_GLOBAL_VIEWS:
        raise ReconstructionValidationError("Could not bind all 40 authoritative SfM workspace images.")

    views: list[CameraBridgeView] = []
    for ordinal, (source_index, provenance) in enumerate(zip(selected_sources, selected_provenance, strict=True)):
        if provenance == "global_sfm_anchor":
            view = global_by_source.get(source_index)
            if view is None:
                raise ReconstructionValidationError(f"Selected global source {source_index} has no SfM anchor.")
        elif provenance == "localized_extra_observation":
            evidence_path = localization_dir / "localization_evidence" / f"candidate_{source_index:08d}.json"
            frame_path = localization_dir / "candidate_frames" / f"candidate_{source_index:08d}.jpg"
            if not evidence_path.is_file() or not frame_path.is_file():
                raise ReconstructionValidationError(
                    f"Localized source {source_index} is missing its pose evidence or materialized frame."
                )
            evidence = _read_json(evidence_path)
            if not bool(evidence.get("accepted")) or bool(evidence.get("is_global_anchor")):
                raise ReconstructionValidationError(f"Localized source {source_index} is not accepted extra-view evidence.")
            if int(evidence.get("source_sequence_index", -1)) != source_index:
                raise ReconstructionValidationError(f"Localization evidence source index mismatch for {source_index}.")
            landmark_ids = evidence.get("matched_landmark_ids", [])
            feature_xy = evidence.get("matched_feature_xy", [])
            if not isinstance(landmark_ids, Sequence) or isinstance(landmark_ids, (str, bytes)):
                raise ReconstructionValidationError("matched_landmark_ids must be an array.")
            if not isinstance(feature_xy, Sequence) or isinstance(feature_xy, (str, bytes)):
                raise ReconstructionValidationError("matched_feature_xy must be an array.")
            point_ids = tuple(_point3d_id(str(value)) for value in landmark_ids)
            xy = tuple(
                tuple(_float_list(value, field_name="matched_feature_xy", length=2))  # type: ignore[arg-type]
                for value in feature_xy
            )
            pose = _pose(evidence.get("T_reconstruction_from_camera"), field_name=f"localized_pose[{source_index}]")
            view = CameraBridgeView(
                source_sequence_index=source_index,
                provenance=provenance,
                image_name=frame_path.name,
                image_path=str(frame_path.resolve()),
                camera_id=shared_camera_id,
                backend_image_id=None,
                T_reconstruction_from_camera=pose,
                matched_point3d_ids=point_ids,
                matched_feature_xy=xy,
            )
        else:
            raise ReconstructionValidationError(f"Unexpected I4.1.7 pose provenance: {provenance}")

        if not np.allclose(view.T_reconstruction_from_camera, diag_matrices[ordinal], rtol=0.0, atol=1e-9):
            raise ReconstructionValidationError(
                f"I4.1.7 pose evidence disagrees with orbit diagnostics for source {source_index}."
            )
        views.append(view)

    image_names = [item.image_name for item in views]
    if len(set(image_names)) != len(image_names):
        raise ReconstructionValidationError("A1 image basenames must be unique across all 68 views.")

    return CameraBridgeInputs(
        i4_1_7_dir=root,
        i4_1_7_summary=summary_path,
        orbit_diagnostics=diagnostics_path,
        sfm_report=sfm_report_path,
        native_sparse_model=native_sparse_model,
        localization_dir=localization_dir,
        selected_model_id=selected_model_id,
        shared_camera_id=shared_camera_id,
        views=tuple(views),
        camera_model_payload=dict(camera_model),
        native_workspace_root=workspace_root,
    )


def run_pycolmap_bridge_worker(
    python_executable: str | Path,
    worker_script: str | Path,
    request: Mapping[str, Any],
    *,
    request_path: Path,
    result_path: Path,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    executable = str(Path(python_executable).expanduser().resolve())
    script = Path(worker_script).expanduser().resolve()
    if not Path(executable).is_file():
        raise ReconstructionValidationError(f"PyCOLMAP Python executable is missing: {executable}")
    if not script.is_file():
        raise ReconstructionValidationError(f"A1 bridge worker script is missing: {script}")
    if timeout_s <= 0 or not isfinite(float(timeout_s)):
        raise ReconstructionValidationError("A1 worker timeout must be finite and positive.")
    _write_json(request_path, request)
    try:
        completed = subprocess.run(
            [executable, str(script), "--request", str(request_path), "--result", str(result_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
        )
    except subprocess.TimeoutExpired as exc:
        raise ReconstructionValidationError(
            f"A1 external PyCOLMAP bridge worker timed out after {float(timeout_s):.3f}s."
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ReconstructionValidationError(
            f"A1 external PyCOLMAP bridge worker failed with code {completed.returncode}: {detail}"
        )
    if not result_path.is_file():
        raise ReconstructionValidationError("A1 external worker did not emit its result JSON.")
    result = _read_json(result_path)
    if result.get("status") != "ok":
        raise ReconstructionValidationError(f"A1 external worker rejected the request: {result.get('error', '')}")
    return result


@dataclass(slots=True)
class _TextImage:
    header_tokens: list[str]
    points_tokens: list[str]

    @property
    def image_id(self) -> int:
        return int(self.header_tokens[0])

    @property
    def camera_id(self) -> int:
        return int(self.header_tokens[8])

    @property
    def name(self) -> str:
        return self.header_tokens[9]


@dataclass(slots=True)
class _TextPoint3D:
    prefix_tokens: list[str]
    track_tokens: list[str]

    @property
    def point3d_id(self) -> int:
        return int(self.prefix_tokens[0])


def _data_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise ReconstructionValidationError(f"COLMAP text model file is missing: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _parse_images_txt(path: Path) -> list[_TextImage]:
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
    data_indices = [idx for idx, line in enumerate(lines) if line.strip() and not line.lstrip().startswith("#")]
    if len(data_indices) % 2 != 0:
        raise ReconstructionValidationError("COLMAP images.txt must contain two data lines per image.")
    records: list[_TextImage] = []
    for offset in range(0, len(data_indices), 2):
        header = lines[data_indices[offset]].strip().split(maxsplit=9)
        points = lines[data_indices[offset + 1]].strip().split()
        if len(header) != 10:
            raise ReconstructionValidationError("COLMAP images.txt image header is malformed.")
        if len(points) % 3 != 0:
            raise ReconstructionValidationError("COLMAP images.txt POINTS2D payload is malformed.")
        records.append(_TextImage(header_tokens=header, points_tokens=points))
    return records


def _parse_points3d_txt(path: Path) -> list[_TextPoint3D]:
    result: list[_TextPoint3D] = []
    for line in _data_lines(path):
        tokens = line.split()
        if len(tokens) < 8 or (len(tokens) - 8) % 2 != 0:
            raise ReconstructionValidationError("COLMAP points3D.txt line is malformed.")
        result.append(_TextPoint3D(prefix_tokens=tokens[:8], track_tokens=tokens[8:]))
    return result


def _write_derived_text_model(
    *,
    native_text_dir: Path,
    output_sparse_dir: Path,
    inputs: CameraBridgeInputs,
) -> tuple[dict[int, int], dict[str, Any]]:
    """Append only the 28 fixed localized observations to the native text model."""

    cameras_path = native_text_dir / "cameras.txt"
    images_path = native_text_dir / "images.txt"
    points_path = native_text_dir / "points3D.txt"
    camera_lines = _data_lines(cameras_path)
    images = _parse_images_txt(images_path)
    points = _parse_points3d_txt(points_path)

    if len(images) != MVS_CAMERA_BRIDGE_EXPECTED_GLOBAL_VIEWS:
        raise ReconstructionValidationError(
            f"Native sparse text contains {len(images)} images, expected 40 authoritative anchors."
        )
    native_image_by_id = {item.image_id: item for item in images}
    if len(native_image_by_id) != len(images):
        raise ReconstructionValidationError("Native COLMAP image IDs are not unique.")
    point_by_id = {item.point3d_id: item for item in points}
    if len(point_by_id) != len(points):
        raise ReconstructionValidationError("Native COLMAP point3D IDs are not unique.")

    native_names = {item.name for item in images}
    for view in inputs.global_views:
        if view.backend_image_id not in native_image_by_id:
            raise ReconstructionValidationError(
                f"Authoritative backend image ID {view.backend_image_id} is absent from the native sparse model."
            )
        native = native_image_by_id[int(view.backend_image_id)]
        if native.name != view.image_name:
            raise ReconstructionValidationError(
                f"Native sparse image name changed for backend image {view.backend_image_id}: {native.name} != {view.image_name}"
            )
        if native.camera_id != inputs.shared_camera_id:
            raise ReconstructionValidationError("Native sparse image does not use the authoritative shared camera ID.")

    max_image_id = max(native_image_by_id) if native_image_by_id else 0
    assigned_ids: dict[int, int] = {}
    appended_observations = 0
    for ordinal, view in enumerate(inputs.localized_views, start=1):
        if view.image_name in native_names:
            raise ReconstructionValidationError(f"Localized image name collides with native sparse model: {view.image_name}")
        image_id = max_image_id + ordinal
        assigned_ids[view.source_sequence_index] = image_id
        camera_from_reconstruction = internal_pose_to_colmap_camera_from_reconstruction(
            view.T_reconstruction_from_camera
        )
        quaternion = rotation_matrix_to_quaternion_wxyz(camera_from_reconstruction[:3, :3])
        translation = camera_from_reconstruction[:3, 3]
        header = [
            str(image_id),
            *(format(float(value), ".17g") for value in quaternion),
            *(format(float(value), ".17g") for value in translation),
            str(inputs.shared_camera_id),
            view.image_name,
        ]
        point_tokens: list[str] = []
        for point2d_idx, (point_id, xy) in enumerate(zip(view.matched_point3d_ids, view.matched_feature_xy, strict=True)):
            point = point_by_id.get(point_id)
            if point is None:
                raise ReconstructionValidationError(
                    f"Localized source {view.source_sequence_index} references missing native point3D {point_id}."
                )
            point_tokens.extend((format(xy[0], ".17g"), format(xy[1], ".17g"), str(point_id)))
            point.track_tokens.extend((str(image_id), str(point2d_idx)))
            appended_observations += 1
        images.append(_TextImage(header_tokens=header, points_tokens=point_tokens))

    output_sparse_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(cameras_path, output_sparse_dir / "cameras.txt")
    with (output_sparse_dir / "images.txt").open("w", encoding="utf-8") as handle:
        handle.write("# I4.1.9A A1 derived sparse model: native cameras/geometry + fixed localized observations\n")
        handle.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        handle.write("# POINTS2D[] as (X Y POINT3D_ID)\n")
        handle.write(f"# Number of images: {len(images)}\n")
        for item in sorted(images, key=lambda record: record.image_id):
            handle.write(" ".join(item.header_tokens) + "\n")
            handle.write(" ".join(item.points_tokens) + "\n")
    with (output_sparse_dir / "points3D.txt").open("w", encoding="utf-8") as handle:
        handle.write("# I4.1.9A A1 preserves native 3D point attributes and appends only extra-view track observations\n")
        handle.write("# POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID POINT2D_IDX)\n")
        handle.write(f"# Number of points: {len(points)}\n")
        for point in points:
            handle.write(" ".join(point.prefix_tokens + point.track_tokens) + "\n")

    return assigned_ids, {
        "native_camera_lines": len(camera_lines),
        "native_registered_images": MVS_CAMERA_BRIDGE_EXPECTED_GLOBAL_VIEWS,
        "localized_registered_images": len(inputs.localized_views),
        "derived_registered_images": len(images),
        "native_points3D": len(points),
        "appended_localized_observations": appended_observations,
    }


def _stage_images(inputs: CameraBridgeInputs, output_images_dir: Path) -> dict[str, str]:
    output_images_dir.mkdir(parents=True, exist_ok=False)
    digests: dict[str, str] = {}
    for view in inputs.views:
        source = Path(view.image_path)
        if not source.is_file():
            raise ReconstructionValidationError(f"A1 source image disappeared: {source}")
        destination = output_images_dir / view.image_name
        if destination.exists():
            raise ReconstructionValidationError(f"A1 refuses duplicate staged image name: {view.image_name}")
        shutil.copy2(source, destination)
        source_digest = sha256_file(source)
        destination_digest = sha256_file(destination)
        if source_digest != destination_digest:
            raise ReconstructionValidationError(f"A1 staged image checksum mismatch: {view.image_name}")
        digests[view.image_name] = destination_digest
    return digests


def _pose_fingerprint(views: Iterable[CameraBridgeView]) -> str:
    payload = [
        {
            "source_sequence_index": item.source_sequence_index,
            "provenance": item.provenance,
            "camera_id": item.camera_id,
            "pose": item.T_reconstruction_from_camera.tolist(),
        }
        for item in sorted(views, key=lambda value: value.source_sequence_index)
    ]
    return _canonical_sha256(payload)


def build_house_mvs_camera_bridge(
    *,
    i4_1_7_dir: str | Path,
    output_dir: str | Path,
    pycolmap_python: str | Path,
    worker_script: str | Path,
    a0_report: str | Path | None = None,
    worker_timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Freeze, bridge, and externally validate the exact 68-view House camera set."""

    output = Path(output_dir).expanduser().resolve()
    _require_clean_output(output)

    if a0_report is not None:
        a0_path = Path(a0_report).expanduser().resolve()
        a0 = _read_json(a0_path)
        if a0.get("phase") != "I4.1.9A-A0" or a0.get("status") != "pass":
            raise ReconstructionValidationError("A1 requires a successful A0 MVS preflight report.")
        if bool(a0.get("heavy_mvs_started")) or bool(a0.get("sfm_rerun")) or bool(a0.get("bundle_adjustment_rerun")):
            raise ReconstructionValidationError("A0 evidence is inconsistent with a read-only preflight.")

    inputs = load_house_camera_bridge_inputs(i4_1_7_dir)

    evidence_dir = output / "evidence"
    worker_dir = output / "worker"
    native_text_dir = output / "native_sparse_text"
    derived_sparse_dir = output / "sparse"
    images_dir = output / "images"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    worker_dir.mkdir(parents=True, exist_ok=True)

    frozen_dir = evidence_dir / "frozen_authority"
    frozen_dir.mkdir(parents=True, exist_ok=False)
    frozen_paths = {
        "i4_1_7_summary": shutil.copy2(inputs.i4_1_7_summary, frozen_dir / "i4_1_7_summary.json"),
        "orbit_diagnostics": shutil.copy2(inputs.orbit_diagnostics, frozen_dir / "i4_1_7_orbit_diagnostics.npz"),
        "real_sfm_report": shutil.copy2(inputs.sfm_report, frozen_dir / "real_sfm_report.json"),
    }
    localization_snapshot = frozen_dir / "localization_evidence"
    localization_snapshot.mkdir()
    for view in inputs.localized_views:
        source = inputs.localization_dir / "localization_evidence" / f"candidate_{view.source_sequence_index:08d}.json"
        shutil.copy2(source, localization_snapshot / source.name)

    export_request = {
        "schema_version": MVS_CAMERA_BRIDGE_SCHEMA_VERSION,
        "action": "export_native_text",
        "native_model": str(inputs.native_sparse_model),
        "output_text_model": str(native_text_dir),
        "expected_registered_images": MVS_CAMERA_BRIDGE_EXPECTED_GLOBAL_VIEWS,
        "forbid_sfm": True,
        "forbid_bundle_adjustment": True,
        "forbid_dense_mvs": True,
    }
    export_result = run_pycolmap_bridge_worker(
        pycolmap_python,
        worker_script,
        export_request,
        request_path=worker_dir / "export_request.json",
        result_path=worker_dir / "export_result.json",
        timeout_s=worker_timeout_s,
    )

    assigned_ids, model_stats = _write_derived_text_model(
        native_text_dir=native_text_dir,
        output_sparse_dir=derived_sparse_dir,
        inputs=inputs,
    )
    image_digests = _stage_images(inputs, images_dir)

    view_manifest = []
    for view in inputs.views:
        image_id = view.backend_image_id if view.backend_image_id is not None else assigned_ids[view.source_sequence_index]
        view_manifest.append(
            {
                **view.as_dict(),
                "derived_image_id": int(image_id),
                "staged_sha256": image_digests[view.image_name],
            }
        )

    authority_manifest = {
        "schema_version": MVS_CAMERA_BRIDGE_SCHEMA_VERSION,
        "phase": "I4.1.9A-A1",
        "status": "bridge_constructed_pending_external_validation",
        "source_i4_1_7_dir": str(inputs.i4_1_7_dir),
        "native_sparse_model": str(inputs.native_sparse_model),
        "selected_model_id": inputs.selected_model_id,
        "camera_model": dict(inputs.camera_model_payload),
        "camera_intrinsics_fingerprint": _canonical_sha256(dict(inputs.camera_model_payload)),
        "camera_pose_fingerprint_68": _pose_fingerprint(inputs.views),
        "counts": {
            "global_sfm_anchor": len(inputs.global_views),
            "localized_extra_observation": len(inputs.localized_views),
            "total_registered_views": len(inputs.views),
        },
        "views": view_manifest,
        "model_stats": model_stats,
        "authority": {
            "single_authoritative_sfm_frame": True,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "pose_refinement": False,
            "camera_intrinsics_modified": False,
            "camera_geometry_modified": False,
            "house_k_txt_consumed": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "heavy_mvs_started": False,
            "image_undistortion_started": False,
            "patch_match_started": False,
            "stereo_fusion_started": False,
        },
        "frozen_evidence_sha256": {
            **{key: sha256_file(Path(value)) for key, value in frozen_paths.items()},
            **{
                f"localization/{path.name}": sha256_file(path)
                for path in sorted(localization_snapshot.glob("candidate_*.json"))
            },
        },
    }
    manifest_path = _write_json(evidence_dir / "a1_camera_authority_manifest.json", authority_manifest)

    validation_request = {
        "schema_version": MVS_CAMERA_BRIDGE_SCHEMA_VERSION,
        "action": "validate_derived_model",
        "native_model": str(inputs.native_sparse_model),
        "derived_model": str(derived_sparse_dir),
        "authority_manifest": str(manifest_path),
        "expected_native_registered_images": MVS_CAMERA_BRIDGE_EXPECTED_GLOBAL_VIEWS,
        "expected_derived_registered_images": MVS_CAMERA_BRIDGE_EXPECTED_TOTAL_VIEWS,
        "forbid_sfm": True,
        "forbid_bundle_adjustment": True,
        "forbid_dense_mvs": True,
    }
    validation_result = run_pycolmap_bridge_worker(
        pycolmap_python,
        worker_script,
        validation_request,
        request_path=worker_dir / "validate_request.json",
        result_path=worker_dir / "validate_result.json",
        timeout_s=worker_timeout_s,
    )

    passed = bool(validation_result.get("validation_pass"))
    summary = {
        "schema_version": MVS_CAMERA_BRIDGE_SCHEMA_VERSION,
        "phase": "I4.1.9A-A1",
        "status": "pass" if passed else "fail",
        "counts": authority_manifest["counts"],
        "model_stats": model_stats,
        "runtime": {
            "pycolmap_version": export_result.get("pycolmap_version"),
            "native_registered_images": export_result.get("registered_images"),
            "native_points3D": export_result.get("points3D"),
        },
        "validation": validation_result,
        "artifacts": {
            "workspace": str(output),
            "images": str(images_dir),
            "sparse": str(derived_sparse_dir),
            "authority_manifest": str(manifest_path),
            "worker_validation": str(worker_dir / "validate_result.json"),
        },
        "authority": authority_manifest["authority"],
        "next": (
            "A1 PASS authorizes only the bounded A2 MVS execution design/run. "
            "It does not authorize metric scale, measurement authority, mesh generation, or any camera re-estimation."
        ),
    }
    _write_json(output / "i4_1_9a_a1_summary.json", summary)
    return summary
