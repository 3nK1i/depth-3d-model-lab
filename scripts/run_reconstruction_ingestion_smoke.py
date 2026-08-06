#!/usr/bin/env python3
"""Non-UI Phase 8.2 ingestion, validation, sampling, and keyframe smoke."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from core.hardware import detect_hardware
from core.hardware.gpu_info import AcceleratorBackend, HardwareInfo
from core.reconstruction import (
    FrameQualityPolicy,
    FrameSamplingMode,
    FrameSamplingPolicy,
    ImageSequenceReader,
    KeyframeSelectionPolicy,
    RecordedVideoReader,
    ReconstructionEventRecorder,
    ReconstructionIngestionPipeline,
    ReconstructionQualityProfile,
    new_reconstruction_session_id,
    resolve_ingestion_cache_policy,
    resolve_reconstruction_profile,
)


def _profile(text: str) -> ReconstructionQualityProfile:
    normalized = text.strip().casefold().replace("-", "_").replace(" ", "_")
    return ReconstructionQualityProfile(normalized)


def _simulated_hardware(name: str, total_gib: float, free_gib: float, capability=(7, 5)) -> HardwareInfo:
    return HardwareInfo(
        backend=AcceleratorBackend.CUDA,
        device="cuda:0",
        device_index=0,
        name=name,
        total_memory_bytes=int(total_gib * 1024**3),
        free_memory_bytes=int(free_gib * 1024**3),
        compute_capability=capability,
        supports_fp16=True,
        supports_bf16=capability[0] >= 8,
        torch_available=True,
        torch_cuda_available=True,
        torch_version="simulated",
        torch_cuda_version="simulated",
        compiled_architectures=(f"sm_{capability[0]}{capability[1]}",),
        torch_build_compatible=True,
    )


def _synthetic_frame(index: int, *, width: int = 320, height: int = 240) -> np.ndarray:
    yy, xx = np.indices((height, width))
    pattern = (((xx // 12 + yy // 12 + index) % 2) * 170 + 40).astype(np.uint8)
    image = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(
        image,
        (25 + index * 8, 45),
        (100 + index * 8, 140),
        (30 + index * 7, 210, 90),
        -1,
    )
    cv2.putText(
        image,
        f"F{index:02d}",
        (15, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return image


def _build_synthetic_sequence(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frames = [_synthetic_frame(index) for index in range(12)]
    frames[5] = frames[4].copy()  # exact duplicate
    frames[8] = np.full_like(frames[8], 127)  # intentionally low quality
    for index, image in enumerate(frames):
        if not cv2.imwrite(str(directory / f"frame{index + 1}.png"), image):
            raise RuntimeError("Could not write synthetic smoke frame.")
    (directory / "frame13.png").write_bytes(b"intentional corrupt image")


def _contact_sheet(images: tuple, output_path: Path) -> None:
    if not images:
        return
    tiles = []
    for index, item in enumerate(images):
        tile = item.image.copy()
        cv2.putText(
            tile,
            f"KF {index}",
            (8, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            f"KF {index}",
            (8, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    target_height = min(tile.shape[0] for tile in tiles)
    resized = [
        cv2.resize(tile, (round(tile.shape[1] * target_height / tile.shape[0]), target_height))
        for tile in tiles
    ]
    sheet = np.concatenate(resized, axis=1)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError("Could not write keyframe contact sheet.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Image directory or recorded video.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/phase8_2_ingestion_smoke"),
    )
    parser.add_argument(
        "--profile",
        default="balanced",
        choices=("efficient", "balanced", "high_quality"),
    )
    parser.add_argument("--target-fps", type=float, default=5.0)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = _profile(args.profile)

    temporary: tempfile.TemporaryDirectory[str] | None = None
    input_path = args.input
    if input_path is None:
        temporary = tempfile.TemporaryDirectory(prefix="phase8_2_smoke_")
        input_path = Path(temporary.name) / "sequence"
        _build_synthetic_sequence(input_path)
    input_path = input_path.expanduser().resolve()

    if input_path.is_dir():
        source = ImageSequenceReader.from_directory(input_path, frame_rate_hz=10.0)
    elif input_path.is_file():
        source = RecordedVideoReader(input_path)
    else:
        raise SystemExit(f"Input does not exist: {input_path}")

    hardware = detect_hardware()
    resolved = resolve_reconstruction_profile(hardware, requested)
    cache_policy = resolve_ingestion_cache_policy(resolved.memory_policy)
    session_id = new_reconstruction_session_id()
    events = ReconstructionEventRecorder(session_id)
    pipeline = ReconstructionIngestionPipeline(
        session_id=session_id,
        source=source,
        sampling_policy=FrameSamplingPolicy(
            mode=FrameSamplingMode.TARGET_FPS,
            target_fps_hz=args.target_fps,
            maximum_sampled_frames=500,
        ),
        keyframe_policy=KeyframeSelectionPolicy(
            minimum_spacing_ns=200_000_000,
            forced_spacing_ns=1_000_000_000,
            minimum_visual_change=0.04,
            maximum_keyframes=min(100, cache_policy.maximum_frames),
        ),
        quality_policy=FrameQualityPolicy(),
        cache_policy=cache_policy,
    )
    result = pipeline.run(events=events)

    report = result.as_dict()
    report["runtime"] = {
        "requested_profile": requested.value,
        "effective_profile": resolved.effective.value,
        "device": resolved.memory_policy.device,
        "warning": resolved.warning,
        "event_count": len(events.events),
    }
    report_path = output_dir / "ingestion_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    contact_sheet_path = output_dir / "keyframes_contact_sheet.png"
    _contact_sheet(result.keyframe_images, contact_sheet_path)

    rtx2060 = resolve_reconstruction_profile(
        _simulated_hardware("NVIDIA GeForce RTX 2060", 6.0, 5.2)
    )
    rtx5090 = resolve_reconstruction_profile(
        _simulated_hardware("NVIDIA GeForce RTX 5090", 32.0, 30.0, (12, 0))
    )

    passed = result.decoded_frames > 0 and result.valid_frames > 0 and len(result.keyframes) > 0
    print(f"Phase 8.2 reconstruction ingestion smoke: {'PASS' if passed else 'FAIL'}")
    print(f"source={source.metadata.kind.value} input={input_path}")
    print(
        "decoded={decoded} valid={valid} rejected={rejected} sampled={sampled} keyframes={keyframes}".format(
            decoded=result.decoded_frames,
            valid=result.valid_frames,
            rejected=result.rejected_frames,
            sampled=result.sampled_frames,
            keyframes=len(result.keyframes),
        )
    )
    print(f"timestamp_origins={json.dumps(result.timestamp_origin_counts, sort_keys=True)}")
    print(
        f"cache={result.cache_stats.current_frames}/{result.cache_policy.maximum_frames} frames, "
        f"peak={result.cache_stats.peak_bytes / 1024**2:.2f} MiB, "
        f"evictions={result.cache_stats.evictions}"
    )
    print(
        f"RTX 2060 6 GB -> {rtx2060.effective.display_name}, "
        f"ingestion cache={resolve_ingestion_cache_policy(rtx2060.memory_policy).maximum_frames} frames"
    )
    print(
        f"RTX 5090 32 GB -> {rtx5090.effective.display_name}, "
        f"ingestion cache={resolve_ingestion_cache_policy(rtx5090.memory_policy).maximum_frames} frames"
    )
    print(f"report={report_path}")
    print(f"keyframes={contact_sheet_path if contact_sheet_path.exists() else 'not generated'}")

    if temporary is not None:
        temporary.cleanup()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
