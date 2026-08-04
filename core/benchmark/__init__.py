"""Registry-driven cross-model benchmark support."""

from core.benchmark.assets import BenchmarkAssetManifest, ensure_benchmark_assets
from core.benchmark.events import EVENT_PREFIX, encode_event, parse_event_line
from core.benchmark.runner import (
    BenchmarkOptions,
    benchmark_model,
    production_benchmark_descriptors,
)

__all__ = [
    "BenchmarkAssetManifest",
    "BenchmarkOptions",
    "EVENT_PREFIX",
    "benchmark_model",
    "encode_event",
    "ensure_benchmark_assets",
    "parse_event_line",
    "production_benchmark_descriptors",
]
