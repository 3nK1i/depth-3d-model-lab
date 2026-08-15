from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from app.da3_canonical_scene_persistence import materialize_da3_canonical_persistent_scene
from core.reconstruction.da3 import DA3_EFFICIENT_MODEL_ID, DA3ResultManifest
from core.reconstruction.da3_canonical_scene import (
    DA3CanonicalSceneIntegrationResult,
    integrate_da3_result_to_canonical_scene,
)
from core.reconstruction.da3_provider import DA3InputManifest, write_da3_input_manifest


def make_phase9_3h_da3_fixture(root: Path, *, request_id: str = "phase9-3h-fixture") -> tuple[Path, Path, DA3ResultManifest]:
    """Create deterministic DA3-shaped evidence without launching the DA3 runtime."""

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    frame_count, height, width = 3, 48, 64
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    depth = np.stack(
        [2.0 + 0.004 * xx + 0.003 * yy + 0.02 * index for index in range(frame_count)]
    ).astype(np.float32)
    confidence = np.ones_like(depth, dtype=np.float32)
    extrinsics = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], frame_count, axis=0)
    for index in range(frame_count):
        extrinsics[index, 0, 3] = -0.03 * index
    intrinsics = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], frame_count, axis=0)
    intrinsics[:, 0, 0] = 55.0
    intrinsics[:, 1, 1] = 55.0
    intrinsics[:, 0, 2] = (width - 1) / 2.0
    intrinsics[:, 1, 2] = (height - 1) / 2.0
    npz_path = root / "da3_output.npz"
    np.savez_compressed(
        npz_path,
        depth=depth,
        confidence=confidence,
        extrinsics_w2c=extrinsics,
        intrinsics=intrinsics,
    )

    image_paths: list[str] = []
    for index in range(frame_count):
        bgr = np.zeros((height, width, 3), dtype=np.uint8)
        bgr[:, :, 0] = 50 + index * 30
        bgr[:, :, 1] = np.linspace(30, 200, width, dtype=np.uint8)[None, :]
        bgr[:, :, 2] = np.linspace(50, 220, height, dtype=np.uint8)[:, None]
        image_path = root / f"frame-{index:04d}.png"
        assert cv2.imwrite(str(image_path), bgr)
        image_paths.append(str(image_path.resolve()))

    input_manifest = DA3InputManifest(
        request_id=request_id,
        image_paths=tuple(image_paths),
        metadata={"phase9_3h_fixture": True},
    )
    input_path = write_da3_input_manifest(root / "input_manifest.json", input_manifest)
    result = DA3ResultManifest(
        request_id=request_id,
        model_id=DA3_EFFICIENT_MODEL_ID,
        model_revision="1" * 40,
        source_revision="2" * 40,
        output_npz_path=str(npz_path.resolve()),
        frame_count=frame_count,
        width=width,
        height=height,
        process_res=392,
        inference_duration_s=0.10,
        peak_vram_mib=100.0,
        gpu_name="fixture-rtx2060-contract",
        metadata={"phase9_3h_fixture": True},
    )
    result_path = root / "result_manifest.json"
    result_path.write_text(json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return input_path, result_path, result


def rgb_frames_for_manifest(input_manifest_path: Path, manifest: DA3ResultManifest) -> tuple[np.ndarray, ...]:
    payload = json.loads(Path(input_manifest_path).read_text(encoding="utf-8"))
    frames: list[np.ndarray] = []
    for item in payload["image_paths"]:
        bgr = cv2.imread(str(item), cv2.IMREAD_COLOR)
        assert bgr is not None
        bgr = cv2.resize(bgr, (manifest.width, manifest.height), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return tuple(frames)


def integrate_phase9_3h_fixture(root: Path, *, session_id: str | None = None) -> tuple[
    DA3CanonicalSceneIntegrationResult,
    Path,
    Path,
]:
    root = Path(root).resolve()
    input_path, result_path, manifest = make_phase9_3h_da3_fixture(root / "evidence")
    sid = session_id or str(uuid4())
    integration = integrate_da3_result_to_canonical_scene(
        manifest,
        session_id=sid,
        request_id=f"phase9-3h-{sid[:8]}",
        rgb_frames=rgb_frames_for_manifest(input_path, manifest),
        pixel_stride=2,
    )
    scene, scene_manifest, _surface_path = materialize_da3_canonical_persistent_scene(
        integration,
        root / "persistent-scene",
        input_manifest_path=input_path,
        result_manifest_path=result_path,
    )
    return integration, scene_manifest, Path(scene_manifest).parent


def write_ready_image_sequence(root: Path) -> Path:
    input_path, _result_path, _manifest = make_phase9_3h_da3_fixture(root / "ready-source")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    return Path(payload["image_paths"][0]).parent
