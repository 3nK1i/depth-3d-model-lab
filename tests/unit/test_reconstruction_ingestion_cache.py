from __future__ import annotations

import numpy as np

from core.reconstruction import (
    BoundedFrameCache,
    IngestionCachePolicy,
    resolve_ingestion_cache_policy,
    resolve_reconstruction_profile,
)
from tests.reconstruction_helpers import make_hardware


def test_bounded_cache_evicts_oldest_without_exceeding_limits() -> None:
    cache = BoundedFrameCache(
        IngestionCachePolicy(maximum_cpu_bytes=600, maximum_frames=2, gpu_staging_bytes=0)
    )
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert cache.put("a", image) == ()
    assert cache.put("b", image) == ()
    assert cache.put("c", image) == ("a",)
    assert cache.get("a") is None
    assert cache.get("c") is not None
    assert cache.stats.current_bytes <= 600
    assert cache.stats.current_frames <= 2
    assert cache.stats.evictions == 1


def test_oversized_frame_is_rejected_without_mutating_cache() -> None:
    cache = BoundedFrameCache(
        IngestionCachePolicy(maximum_cpu_bytes=100, maximum_frames=2, gpu_staging_bytes=0)
    )
    assert cache.put("large", np.zeros((10, 10, 3), dtype=np.uint8)) == ()
    assert cache.stats.current_frames == 0
    assert cache.stats.rejected_oversized_frames == 1


def test_ingestion_cache_policy_adapts_to_rtx2060_and_rtx5090() -> None:
    rtx2060 = resolve_ingestion_cache_policy(
        resolve_reconstruction_profile(
            make_hardware(total_gib=6.0, free_gib=5.2, name="RTX 2060")
        ).memory_policy
    )
    rtx5090 = resolve_ingestion_cache_policy(
        resolve_reconstruction_profile(
            make_hardware(total_gib=32.0, free_gib=30.0, capability=(12, 0), name="RTX 5090")
        ).memory_policy
    )
    assert rtx2060.maximum_frames == 48
    assert rtx5090.maximum_frames == 512
    assert rtx5090.maximum_cpu_bytes > rtx2060.maximum_cpu_bytes
    assert rtx5090.gpu_staging_bytes > rtx2060.gpu_staging_bytes
