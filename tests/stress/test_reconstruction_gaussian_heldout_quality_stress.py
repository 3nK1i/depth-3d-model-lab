from __future__ import annotations

from core.reconstruction import (
    GaussianHeldOutViewMetrics,
    aggregate_gaussian_held_out_metrics,
)


def test_heldout_quality_aggregation_is_stable_under_repeated_serial_values() -> None:
    for iteration in range(250):
        views = tuple(
            GaussianHeldOutViewMetrics(
                frame_id=f"frame-{index}",
                psnr_db=20.0 + index * 0.1 + iteration * 1e-6,
                ssim=0.8 + index * 0.001,
                lpips=0.2 + index * 0.001,
            )
            for index in range(8)
        )
        aggregate = aggregate_gaussian_held_out_metrics(views)
        assert aggregate.measured is True
        assert aggregate.psnr_db is not None and aggregate.psnr_db > 20.0
        assert aggregate.ssim is not None and 0.8 <= aggregate.ssim <= 1.0
        assert aggregate.lpips is not None and aggregate.lpips >= 0.2
