from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "scripts" / "reconstruction_progressive_sfm_emission.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("i6_6b_progressive_stress", HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_interval_policy_is_bounded_across_large_source_counts() -> None:
    helper = _load_helper()
    for count in range(3, 10001):
        interval = helper.mapping_snapshot_frame_interval(count)
        assert interval >= 2
        assert interval <= max(2, count)
        estimated = max(0, (count - 2) // interval)
        assert estimated <= helper.PROGRESSIVE_TARGET_MAPPING_SNAPSHOTS + 1


def test_throttle_never_reemits_same_native_snapshot_token() -> None:
    helper = _load_helper()
    throttle = helper.ProgressiveSnapshotThrottle(min_interval_s=0.0)
    accepted = []
    for index in range(5000):
        token = f"snapshot-{index // 2}"
        if throttle.should_emit(token, now_s=float(index)):
            accepted.append(token)
    assert len(accepted) == 2500
    assert len(accepted) == len(set(accepted))
