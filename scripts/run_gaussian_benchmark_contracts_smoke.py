#!/usr/bin/env python3
"""Dependency-light Phase 8.7D3 benchmark-contract smoke."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    DenseSceneProfile,
    GaussianBackendKind,
    GaussianBenchmarkHardwareProfile,
    GaussianOptimizationProfile,
    NeuralSceneDataset,
    NeuralSceneFrame,
    RealSceneGaussianBenchmarkReport,
    RealSceneGaussianBenchmarkSpec,
    write_real_scene_gaussian_benchmark_report,
    write_real_scene_gaussian_benchmark_spec,
)
from core.reconstruction.scale_reference import MetricGeometryStatus
from core.reconstruction.sfm_contracts import SfmScaleStatus, SfmWorldAlignmentStatus

OUTPUT = Path("/tmp/phase8_7d3_benchmark_contracts")


def _dataset() -> NeuralSceneDataset:
    images = OUTPUT / "images"
    images.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(5):
        image = (images / f"frame_{index:03d}.ppm").resolve()
        image.write_bytes(b"P6\n2 2\n255\n" + bytes([70 + index * 10, 90, 110]) * 4)
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = 0.05 * index
        frames.append(
            NeuralSceneFrame(
                frame_id=f"view-{index}",
                image_path=str(image),
                width=2,
                height=2,
                fx=2.0,
                fy=2.0,
                cx=1.0,
                cy=1.0,
                T_reconstruction_from_camera=pose,
                pose_layer="optimized_keyframe",
                camera_profile_id="phase8-7d3-smoke-camera",
                timestamp_ns=index * 33_333_333,
            )
        )
    return NeuralSceneDataset(
        session_id="phase8-7d3-contract-smoke",
        scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        reconstruction_frame_id="phase8-7d3-reconstruction",
        camera_profile_id="phase8-7d3-smoke-camera",
        scale_profile_id="phase8-7d3-arbitrary-scale",
        metric_geometry_status=MetricGeometryStatus.ARBITRARY,
        scale_status=SfmScaleStatus.ARBITRARY,
        world_alignment_status=SfmWorldAlignmentStatus.ARBITRARY,
        unit_to_metre=None,
        relative_scale_uncertainty=None,
        frames=tuple(frames),
    )


def main() -> int:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    dataset = _dataset()
    spec = RealSceneGaussianBenchmarkSpec(
        benchmark_id="phase8-7d3-contract-smoke",
        source_label="deterministic-contract-smoke",
        dataset=dataset,
        training_frame_ids=("view-0", "view-1", "view-2", "view-3"),
        validation_frame_ids=("view-4",),
        backends=(
            GaussianBackendKind.GSPLAT_EXTERNAL,
            GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL,
        ),
        optimization_profile=GaussianOptimizationProfile.EFFICIENT,
        hardware_profile=GaussianBenchmarkHardwareProfile.RTX_2060_6GB,
        hardware_label="NVIDIA GeForce RTX 2060 6 GB",
    )
    spec_path = write_real_scene_gaussian_benchmark_spec(spec, OUTPUT / "benchmark_spec.json")
    report = RealSceneGaussianBenchmarkReport(spec=spec)
    report_path = write_real_scene_gaussian_benchmark_report(report, OUTPUT / "benchmark_report.json")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["comparison_ready"] is False
    assert payload["spec"]["training_frame_ids"] == ["view-0", "view-1", "view-2", "view-3"]
    assert payload["spec"]["validation_frame_ids"] == ["view-4"]
    assert payload["canonical_geometry_authoritative"] is False
    assert payload["certified_metrology"] is False
    print("Phase 8.7D3 benchmark contracts smoke: PASS")
    print(f"benchmark_fingerprint={spec.benchmark_fingerprint_sha256}")
    print(f"training_views={len(spec.training_frame_ids)} held_out_views={len(spec.validation_frame_ids)}")
    print("metrics_measured=False performance_measured=False")
    print("canonical_geometry_authoritative=False certified_metrology=False")
    print(f"spec={spec_path}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
