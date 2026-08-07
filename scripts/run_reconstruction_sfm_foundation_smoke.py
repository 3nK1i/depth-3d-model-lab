#!/usr/bin/env python3
"""Non-UI Phase 8.3B gauge/workspace/mock-SfM foundation smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from core.reconstruction import (
    BackendOperationContext,
    DeterministicMockSfmBackend,
    FrameQualityPolicy,
    FrameSamplingMode,
    FrameSamplingPolicy,
    ImageSequenceReader,
    IngestionCachePolicy,
    KeyframeSelectionPolicy,
    OfflineSfmRequest,
    ReconstructionIngestionPipeline,
    ReconstructionOperationControl,
    ReconstructionQualityProfile,
    ReconstructionStage,
    materialize_sfm_workspace,
    new_reconstruction_session_id,
    probe_pycolmap_environment,
)


def _synthetic_frame(index: int, *, width: int = 320, height: int = 240):
    import numpy as np

    yy, xx = np.indices((height, width))
    checker = (((xx // 10 + yy // 10 + index) % 2) * 170 + 40).astype(np.uint8)
    image = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)
    cv2.circle(
        image,
        (50 + index * 22, 90 + (index % 2) * 18),
        32,
        (30 + index * 12, 210, 90),
        -1,
    )
    cv2.putText(
        image,
        f"SFM {index:02d}",
        (15, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return image


def _build_sequence(directory: Path, count: int = 8) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        path = directory / f"frame{index + 1}.png"
        if not cv2.imwrite(str(path), _synthetic_frame(index)):
            raise RuntimeError(f"Could not write synthetic frame: {path}")


def _write_mock_ply(result, path: Path) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        "comment Phase 8.3B deterministic mock sparse model",
        "comment arbitrary reconstruction gauge; coordinates are NOT metres",
        f"element vertex {len(result.landmarks)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for landmark in result.landmarks:
        x, y, z = landmark.position_reconstruction
        r, g, b = landmark.color_rgb or (180, 180, 180)
        lines.append(f"{x:.9g} {y:.9g} {z:.9g} {r} {g} {b}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/phase8_3b_sfm_foundation_smoke"),
    )
    parser.add_argument(
        "--sfm-python",
        help=(
            "Optional external reconstruction-environment Python executable. "
            "Example: /home/eiatech/anaconda3/envs/eia-pycolmap-probe/bin/python"
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="phase8_3b_sequence_") as temp:
        sequence = Path(temp) / "sequence"
        _build_sequence(sequence)
        source = ImageSequenceReader.from_directory(sequence, frame_rate_hz=10.0)
        session_id = new_reconstruction_session_id()
        ingestion = ReconstructionIngestionPipeline(
            session_id=session_id,
            source=source,
            sampling_policy=FrameSamplingPolicy(mode=FrameSamplingMode.EVERY_FRAME),
            keyframe_policy=KeyframeSelectionPolicy(
                minimum_spacing_ns=1,
                forced_spacing_ns=1,
                minimum_visual_change=0.0,
                minimum_quality_score=0.0,
                maximum_keyframes=8,
            ),
            quality_policy=FrameQualityPolicy(
                minimum_blur_variance=0.0,
                minimum_contrast_stddev=0.0,
                minimum_mean_luma=0.0,
                maximum_mean_luma=255.0,
                maximum_clipped_fraction=1.0,
                minimum_quality_score=0.0,
            ),
            cache_policy=IngestionCachePolicy(
                maximum_cpu_bytes=64 * 1024 * 1024,
                maximum_frames=2,
                gpu_staging_bytes=0,
            ),
        ).run()
        workspace = materialize_sfm_workspace(
            ingestion,
            output_dir / "workspace",
            overwrite=True,
        )

    request = OfflineSfmRequest(
        session_id=session_id,
        profile=ReconstructionQualityProfile.BALANCED,
        workspace=workspace,
    )
    emitted = []
    context = BackendOperationContext(
        session_id=session_id,
        stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
        profile=ReconstructionQualityProfile.BALANCED,
        control=ReconstructionOperationControl(),
        emit_event=emitted.append,
    )
    backend = DeterministicMockSfmBackend()
    result = backend.run_sparse_reconstruction(request, context)
    backend.shutdown()

    probe_executable = args.sfm_python or sys.executable
    probe = probe_pycolmap_environment(probe_executable)

    report = {
        "phase": "8.3B",
        "real_sfm_executed": False,
        "sfm_result": result.as_dict(),
        "external_pycolmap_probe": probe.as_dict(),
        "event_count": len(emitted),
        "phase82_keyframe_cache_count": len(ingestion.keyframe_images),
        "materialized_keyframe_count": len(workspace.images),
    }
    report_path = output_dir / "sfm_foundation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    ply_path = output_dir / "mock_sparse_arbitrary_scale.ply"
    _write_mock_ply(result, ply_path)

    passed = (
        len(workspace.images) == len(ingestion.keyframes)
        and len(workspace.images) > len(ingestion.keyframe_images)
        and result.gauge.unit_to_metre is None
        and result.statistics.registered_keyframes == len(workspace.images)
        and emitted[-1].progress_fraction == 1.0
    )

    print(f"Phase 8.3B SfM foundation smoke: {'PASS' if passed else 'FAIL'}")
    print(
        f"ingestion_keyframes={len(ingestion.keyframes)} "
        f"bounded_cache={len(ingestion.keyframe_images)} "
        f"materialized_workspace={len(workspace.images)}"
    )
    print(
        f"mock_registered_cameras={len(result.cameras)} "
        f"mock_landmarks={len(result.landmarks)} "
        f"gauge={result.gauge.scale_status.value} unit_to_metre={result.gauge.unit_to_metre}"
    )
    print("real_sfm_executed=False")
    print(
        "pycolmap_probe="
        f"available:{probe.available} version:{probe.pycolmap_version or 'n/a'} "
        f"cuda_devices:{probe.cuda_devices} backend_ready:{probe.backend_ready}"
    )
    print(f"workspace={workspace.workspace_root}")
    print(f"report={report_path}")
    print(f"mock_ply={ply_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
