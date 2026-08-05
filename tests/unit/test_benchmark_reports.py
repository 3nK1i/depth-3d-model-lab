"""Phase 6 benchmark report writer tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from core.benchmark.reports import (
    write_benchmark_csv,
    write_benchmark_summary,
    write_capability_matrix,
    write_json,
)
from core.benchmark.runner import production_benchmark_descriptors
from core.version import __version__


def _result() -> dict[str, object]:
    return {
        "model_id": "model-a",
        "display_name": "Model A",
        "status": "passed",
        "classification": "metric_depth",
        "workload": "image",
        "runtime": {
            "effective_profile": "efficient",
            "device": "cuda:0",
            "precision": "FP16",
        },
        "timing": {
            "load_ms": 10.0,
            "warmup_ms": 5.0,
            "model_latency_ms": {"mean": 20.0},
            "end_to_end_ms": {"mean": 25.0},
            "fps": 40.0,
        },
        "memory": {"peak_reserved_gib": 1.25, "device_used_gib": 1.5},
        "output": {
            "resolution": [640, 384],
            "point_count": 100_000,
            "depth_scale": "metric",
            "is_metric": True,
            "intrinsics_source": "estimated",
            "confidence_kind": "none",
        },
    }


def test_benchmark_reports_are_machine_and_human_readable(tmp_path: Path) -> None:
    result = _result()
    payload = {
        "application": {"version": __version__},
        "hardware": {"name": "Test GPU"},
        "results": [result],
    }

    json_path = write_json(tmp_path / "benchmark.json", payload)
    csv_path = write_benchmark_csv(tmp_path / "benchmark.csv", [result])
    summary_path = write_benchmark_summary(tmp_path / "summary.md", payload)
    matrix_path = write_capability_matrix(
        tmp_path / "matrix.md", production_benchmark_descriptors()
    )

    assert json.loads(json_path.read_text(encoding="utf-8"))["hardware"]["name"] == "Test GPU"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["model_id"] == "model-a"
    assert rows[0]["mean_end_to_end_ms"] == "25.0"
    assert "Model A" in summary_path.read_text(encoding="utf-8")
    matrix = matrix_path.read_text(encoding="utf-8")
    assert "MoGe-2" in matrix
    assert "Estimated — Not Calibrated" in matrix


def test_cancelled_runs_are_visible_in_human_summary(tmp_path: Path) -> None:
    payload = {
        "application": {"version": __version__},
        "hardware": {"name": "Test GPU"},
        "cancelled": True,
        "results": [
            {
                "model_id": "model-a",
                "display_name": "Model A",
                "status": "cancelled",
                "error": "Benchmark cancelled by the user.",
            }
        ],
    }

    summary = write_benchmark_summary(tmp_path / "cancelled.md", payload).read_text(
        encoding="utf-8"
    )

    assert "Cancelled: **1**" in summary
    assert "Benchmark cancelled by the user" in summary
