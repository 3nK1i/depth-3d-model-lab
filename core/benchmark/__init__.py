"""Registry-driven cross-model benchmark support for the Phase 6 release."""

from core.benchmark.assets import BenchmarkAssetManifest, ensure_benchmark_assets
from core.benchmark.runner import (
    BenchmarkOptions,
    benchmark_model,
    production_benchmark_descriptors,
)

__all__ = [
    "BenchmarkAssetManifest",
    "BenchmarkOptions",
    "benchmark_model",
    "ensure_benchmark_assets",
    "production_benchmark_descriptors",
]
