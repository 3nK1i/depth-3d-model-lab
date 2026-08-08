from __future__ import annotations

from core.reconstruction import GaussianPerformanceMetrics


def test_step3_performance_contract_remains_stable_over_repeated_serialization() -> None:
    for index in range(250):
        perf = GaussianPerformanceMetrics(
            peak_training_vram_mib=2000.0 + index,
            peak_render_vram_mib=1500.0 + index,
            training_duration_s=5.0 + index / 100.0,
            render_duration_s=2.0,
            render_frame_count=60,
            render_repeats=3,
            render_fps=30.0,
            measurement_scope="backend_process_tree_nvidia_smi",
            render_measurement_mode="backend_native_warm",
        )
        payload = perf.as_dict()
        assert payload["detailed_measured"] is True
        assert payload["peak_vram_mib"] == 2000.0 + index
