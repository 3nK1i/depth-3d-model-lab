"""Dependency-neutral OfflineSfmResult JSON rehydration helpers.

The Phase 8/9 SfM contracts already provide stable ``as_dict`` payloads, but
several diagnostic scripts historically carried private rehydration copies.
G4B needs to consume a *frozen* authoritative SfM JSON without rerunning
PyCOLMAP, so the small parser lives here as a reusable core helper.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from core.contracts import ImageSize
from core.reconstruction.contracts import InputSourceKind, InputSourceMetadata, TimestampOrigin
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.sfm_contracts import (
    OfflineSfmResult,
    ReconstructionGauge,
    SfmCameraEstimate,
    SfmCameraModelEstimate,
    SfmLandmarkEstimate,
    SfmObservation,
    SfmScaleStatus,
    SfmStageStatistics,
    SfmWorkspaceImage,
    SfmWorkspaceManifest,
    SfmWorldAlignmentStatus,
)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"Offline SfM payload section must be an object: {label}")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReconstructionValidationError(f"Offline SfM payload section must be an array: {label}")
    return value


def _image_size(value: Any, *, label: str) -> ImageSize:
    raw = _mapping(value, label=label)
    return ImageSize(width=int(raw["width"]), height=int(raw["height"]))


def offline_sfm_result_from_dict(payload: Mapping[str, Any]) -> OfflineSfmResult:
    """Rehydrate a validated ``OfflineSfmResult.as_dict()`` payload.

    This parser is intentionally lossless for the persisted SfM authority used
    by the dense-provider pipeline. It does not infer, repair, or normalize a
    different world frame and therefore cannot promote provider-native poses.
    """

    raw = _mapping(payload, label="root")
    try:
        gauge_raw = _mapping(raw["gauge"], label="gauge")
        gauge = ReconstructionGauge(
            scale_status=SfmScaleStatus(str(gauge_raw["scale_status"])),
            alignment_status=SfmWorldAlignmentStatus(str(gauge_raw["alignment_status"])),
            unit_to_metre=(
                None if gauge_raw.get("unit_to_metre") is None else float(gauge_raw["unit_to_metre"])
            ),
            scale_source=str(gauge_raw.get("scale_source", "")),
            relative_scale_uncertainty=(
                None
                if gauge_raw.get("relative_scale_uncertainty") is None
                else float(gauge_raw["relative_scale_uncertainty"])
            ),
        )

        workspace_raw = _mapping(raw["workspace"], label="workspace")
        source_raw = _mapping(workspace_raw["source"], label="workspace.source")
        expected_size_raw = source_raw.get("expected_image_size")
        source = InputSourceMetadata(
            source_id=str(source_raw["source_id"]),
            kind=InputSourceKind(str(source_raw["kind"])),
            display_name=str(source_raw["display_name"]),
            source_uri=str(source_raw["source_uri"]),
            expected_image_size=(
                None
                if expected_size_raw is None
                else _image_size(expected_size_raw, label="workspace.source.expected_image_size")
            ),
            expected_frame_rate_hz=(
                None
                if source_raw.get("expected_frame_rate_hz") is None
                else float(source_raw["expected_frame_rate_hz"])
            ),
            timestamp_origin=TimestampOrigin(str(source_raw.get("timestamp_origin", "unknown"))),
            fingerprint=str(source_raw.get("fingerprint", "")),
            metadata=source_raw.get("metadata", {}),
        )
        workspace_images = tuple(
            SfmWorkspaceImage(
                order=int(item["order"]),
                keyframe_id=str(item["keyframe_id"]),
                frame_id=str(item["frame_id"]),
                source_sequence_index=int(item["source_sequence_index"]),
                timestamp_ns=int(item["timestamp_ns"]),
                image_size=_image_size(item["image_size"], label="workspace.image.image_size"),
                source_locator=str(item["source_locator"]),
                staged_image_path=str(item["staged_image_path"]),
                staged_sha256=str(item["staged_sha256"]),
            )
            for item_raw in _sequence(workspace_raw["images"], label="workspace.images")
            for item in (_mapping(item_raw, label="workspace.image"),)
        )
        paths = _mapping(workspace_raw["paths"], label="workspace.paths")
        workspace = SfmWorkspaceManifest(
            session_id=str(workspace_raw["session_id"]),
            source=source,
            workspace_root=str(workspace_raw["workspace_root"]),
            images_directory=str(paths["images_directory"]),
            database_path=str(paths["database_path"]),
            sparse_directory=str(paths["sparse_directory"]),
            checkpoints_directory=str(paths["checkpoints_directory"]),
            reports_directory=str(paths["reports_directory"]),
            logs_directory=str(paths["logs_directory"]),
            images=workspace_images,
            schema_version=int(workspace_raw.get("schema_version", 1)),
        )

        cameras = tuple(
            SfmCameraEstimate(
                estimate_id=str(item["estimate_id"]),
                camera_id=str(item["camera_id"]),
                keyframe_id=str(item["keyframe_id"]),
                frame_id=str(item["frame_id"]),
                backend_image_id=(
                    None if item.get("backend_image_id") is None else int(item["backend_image_id"])
                ),
                T_reconstruction_from_camera=np.asarray(
                    item["T_reconstruction_from_camera"], dtype=np.float64
                ),
            )
            for item_raw in _sequence(raw["cameras"], label="cameras")
            for item in (_mapping(item_raw, label="camera"),)
        )
        camera_models = tuple(
            SfmCameraModelEstimate(
                camera_id=str(item["camera_id"]),
                backend_camera_id=int(item["backend_camera_id"]),
                backend_model_name=str(item["backend_model_name"]),
                image_size=_image_size(item["image_size"], label="camera_model.image_size"),
                parameters=tuple(float(value) for value in item["parameters"]),
            )
            for item_raw in _sequence(raw.get("camera_models", ()), label="camera_models")
            for item in (_mapping(item_raw, label="camera_model"),)
        )

        landmarks: list[SfmLandmarkEstimate] = []
        for item_raw in _sequence(raw["landmarks"], label="landmarks"):
            item = _mapping(item_raw, label="landmark")
            observations = tuple(
                SfmObservation(
                    observation_id=str(observation["observation_id"]),
                    landmark_id=str(observation["landmark_id"]),
                    keyframe_id=str(observation["keyframe_id"]),
                    feature_xy=tuple(float(value) for value in observation["feature_xy"]),
                    feature_index=(
                        None if observation.get("feature_index") is None else int(observation["feature_index"])
                    ),
                    backend_point2d_index=(
                        None
                        if observation.get("backend_point2d_index") is None
                        else int(observation["backend_point2d_index"])
                    ),
                )
                for observation_raw in _sequence(item["observations"], label="landmark.observations")
                for observation in (_mapping(observation_raw, label="landmark.observation"),)
            )
            color_raw = item.get("color_rgb")
            landmarks.append(
                SfmLandmarkEstimate(
                    landmark_id=str(item["landmark_id"]),
                    track_id=str(item["track_id"]),
                    position_reconstruction=np.asarray(item["position_reconstruction"], dtype=np.float64),
                    observations=observations,
                    color_rgb=(
                        None if color_raw is None else tuple(int(value) for value in color_raw)
                    ),
                )
            )

        stats_raw = _mapping(raw["statistics"], label="statistics")
        statistics = SfmStageStatistics(
            input_keyframes=int(stats_raw["input_keyframes"]),
            registered_keyframes=int(stats_raw["registered_keyframes"]),
            landmarks=int(stats_raw["landmarks"]),
            observations=int(stats_raw["observations"]),
            elapsed_ns=int(stats_raw["elapsed_ns"]),
            mean_reprojection_error_px=(
                None
                if stats_raw.get("mean_reprojection_error_px") is None
                else float(stats_raw["mean_reprojection_error_px"])
            ),
        )
        return OfflineSfmResult(
            session_id=str(raw["session_id"]),
            backend_id=str(raw["backend_id"]),
            backend_version=str(raw["backend_version"]),
            gauge=gauge,
            cameras=cameras,
            camera_models=camera_models,
            landmarks=tuple(landmarks),
            statistics=statistics,
            workspace=workspace,
            warnings=tuple(str(value) for value in raw.get("warnings", ())),
            metadata=raw.get("metadata", {}),
            schema_version=int(raw.get("schema_version", 1)),
        )
    except ReconstructionValidationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ReconstructionValidationError(f"Invalid offline SfM JSON payload: {exc}") from exc


__all__ = ["offline_sfm_result_from_dict"]
