"""Durable workspace materialization for selected Phase 8.2 keyframes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil

import cv2

from core.reconstruction.contracts import InputSourceKind
from core.reconstruction.errors import ReconstructionValidationError, SfmWorkspaceError
from core.reconstruction.ingestion import ReconstructionIngestionResult
from core.reconstruction.sfm_contracts import SfmWorkspaceImage, SfmWorkspaceManifest

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_component(value: str) -> str:
    cleaned = _SAFE_NAME.sub("_", value.strip()).strip("._")
    return (cleaned or "keyframe")[:80]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(manifest: SfmWorkspaceManifest) -> Path:
    path = Path(manifest.workspace_root) / "manifest.json"
    path.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _records_for_selected_keyframes(result: ReconstructionIngestionResult):
    by_keyframe = {record.keyframe_id: record for record in result.records if record.keyframe_id}
    selected = sorted(result.keyframes, key=lambda item: item.selection_sequence)
    records = []
    for keyframe in selected:
        record = by_keyframe.get(keyframe.keyframe_id)
        if record is None:
            raise SfmWorkspaceError(
                f"Selected keyframe has no ingestion record: {keyframe.keyframe_id}"
            )
        if not record.frame.is_valid or not record.sampled:
            raise SfmWorkspaceError(
                f"Selected keyframe is not backed by a valid sampled frame: {keyframe.keyframe_id}"
            )
        records.append((keyframe, record))
    if len(records) < 2:
        raise SfmWorkspaceError("Offline SfM requires at least two selected keyframes.")
    return records


def _materialize_image_sequence(records, images_dir: Path) -> list[SfmWorkspaceImage]:
    output: list[SfmWorkspaceImage] = []
    for order, (keyframe, record) in enumerate(records):
        source = Path(record.source_locator).expanduser()
        if not source.is_file():
            raise SfmWorkspaceError(
                f"Keyframe source image is no longer available: {record.source_locator}"
            )
        suffix = source.suffix.casefold() or ".png"
        filename = f"kf_{order:06d}_{_safe_component(keyframe.keyframe_id)}{suffix}"
        destination = images_dir / filename
        shutil.copyfile(source, destination)
        output.append(
            SfmWorkspaceImage(
                order=order,
                keyframe_id=keyframe.keyframe_id,
                frame_id=record.frame.frame_id,
                source_sequence_index=record.frame.sequence_index,
                timestamp_ns=record.frame.timestamp_ns,
                image_size=record.frame.image_size,
                source_locator=record.source_locator,
                staged_image_path=f"images/{filename}",
                staged_sha256=_sha256_file(destination),
            )
        )
    return output


def _materialize_recorded_video(
    result: ReconstructionIngestionResult,
    records,
    images_dir: Path,
) -> list[SfmWorkspaceImage]:
    video_path = Path(result.source.source_uri).expanduser()
    if not video_path.is_file():
        raise SfmWorkspaceError(f"Recorded video is no longer available: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise SfmWorkspaceError(f"Could not open recorded video: {video_path}")
    wanted = {record.frame.sequence_index: (order, keyframe, record) for order, (keyframe, record) in enumerate(records)}
    output_by_order: dict[int, SfmWorkspaceImage] = {}
    max_index = max(wanted)
    index = 0
    try:
        while index <= max_index:
            ok, image = capture.read()
            if not ok:
                break
            item = wanted.get(index)
            if item is not None:
                order, keyframe, record = item
                filename = f"kf_{order:06d}_{_safe_component(keyframe.keyframe_id)}.png"
                destination = images_dir / filename
                if not cv2.imwrite(str(destination), image):
                    raise SfmWorkspaceError(
                        f"Could not materialize video keyframe {keyframe.keyframe_id}."
                    )
                output_by_order[order] = SfmWorkspaceImage(
                    order=order,
                    keyframe_id=keyframe.keyframe_id,
                    frame_id=record.frame.frame_id,
                    source_sequence_index=record.frame.sequence_index,
                    timestamp_ns=record.frame.timestamp_ns,
                    image_size=record.frame.image_size,
                    source_locator=record.source_locator,
                    staged_image_path=f"images/{filename}",
                    staged_sha256=_sha256_file(destination),
                )
            index += 1
    finally:
        capture.release()
    if len(output_by_order) != len(records):
        missing = sorted(set(range(len(records))) - set(output_by_order))
        raise SfmWorkspaceError(
            f"Could not decode all selected video keyframes; missing workspace orders: {missing}."
        )
    return [output_by_order[index] for index in range(len(records))]



def validate_sfm_workspace_integrity(manifest: SfmWorkspaceManifest) -> None:
    """Verify materialized SfM images still match their recorded SHA-256 digests."""

    if not isinstance(manifest, SfmWorkspaceManifest):
        raise TypeError("manifest must be an SfmWorkspaceManifest.")
    root = Path(manifest.workspace_root).expanduser().resolve()
    if not root.is_dir():
        raise SfmWorkspaceError(f"SfM workspace does not exist: {root}")
    seen: set[Path] = set()
    for item in manifest.images:
        staged = (root / item.staged_image_path).resolve()
        try:
            staged.relative_to(root)
        except ValueError as exc:
            raise SfmWorkspaceError(
                f"Materialized SfM image escapes the workspace: {item.staged_image_path}"
            ) from exc
        if staged in seen:
            raise SfmWorkspaceError(
                f"Duplicate materialized SfM image path: {item.staged_image_path}"
            )
        seen.add(staged)
        if not staged.is_file():
            raise SfmWorkspaceError(
                f"Materialized SfM image is missing: {item.staged_image_path}"
            )
        digest = _sha256_file(staged)
        if digest != item.staged_sha256:
            raise SfmWorkspaceError(
                f"Materialized SfM image checksum changed: {item.staged_image_path}"
            )

def materialize_sfm_workspace(
    result: ReconstructionIngestionResult,
    workspace_root: str | Path,
    *,
    overwrite: bool = False,
) -> SfmWorkspaceManifest:
    """Materialize every selected keyframe into a durable SfM workspace."""

    if not isinstance(result, ReconstructionIngestionResult):
        raise TypeError("result must be a ReconstructionIngestionResult.")
    root = Path(workspace_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise SfmWorkspaceError(f"SfM workspace is not empty: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    images_dir = root / "images"
    for relative in ("images", "sparse", "checkpoints", "reports", "logs"):
        (root / relative).mkdir(parents=True, exist_ok=True)

    records = _records_for_selected_keyframes(result)
    if result.source.kind is InputSourceKind.IMAGE_SEQUENCE:
        images = _materialize_image_sequence(records, images_dir)
    elif result.source.kind is InputSourceKind.RECORDED_VIDEO:
        images = _materialize_recorded_video(result, records, images_dir)
    else:
        raise ReconstructionValidationError(
            "Phase 8.3 offline workspace supports image sequences and recorded videos only."
        )

    session_ids = {record.frame.session_id for _, record in records}
    if len(session_ids) != 1:
        raise SfmWorkspaceError("Selected keyframes do not share one reconstruction session.")
    manifest = SfmWorkspaceManifest(
        session_id=next(iter(session_ids)),
        source=result.source,
        workspace_root=str(root),
        images_directory="images",
        database_path="database.db",
        sparse_directory="sparse",
        checkpoints_directory="checkpoints",
        reports_directory="reports",
        logs_directory="logs",
        images=tuple(images),
    )
    _write_manifest(manifest)
    return manifest
