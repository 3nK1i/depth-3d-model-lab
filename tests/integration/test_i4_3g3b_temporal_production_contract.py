from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.temporal_provider_production_qualification import (
    I43G3B_PROVIDER_IMPLEMENTATIONS,
    I43G3B_SCENARIOS,
)

ROOT = Path(__file__).resolve().parents[2]


def test_g3b_frozen_temporal_matrix_and_runner_cli_contract() -> None:
    assert I43G3B_SCENARIOS == (
        "g3_temporal_static_sweep",
        "g3_temporal_motion",
        "g3_temporal_low_texture",
    )
    assert I43G3B_PROVIDER_IMPLEMENTATIONS == ("rolling_depth", "video_depth_anything")

    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_i4_3g3b_temporal_scenario.py"), "--help"],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "preflight" in completed.stdout
    assert "production-native" in completed.stdout
    assert "--i4-3g3a-snapshot" in completed.stdout
    assert "--i4-3g3a-sha256" in completed.stdout
    assert "--scenario" in completed.stdout


def test_g3b_temporal_order_is_read_from_materialized_sfm_workspace() -> None:
    from scripts.run_i4_3g3b_temporal_scenario import _validate_temporal_workspace_order

    workspace = SimpleNamespace(
        images=(
            SimpleNamespace(source_sequence_index=2, timestamp_ns=100),
            SimpleNamespace(source_sequence_index=5, timestamp_ns=200),
            SimpleNamespace(source_sequence_index=9, timestamp_ns=300),
        )
    )
    assert _validate_temporal_workspace_order(workspace) == ([2, 5, 9], [100, 200, 300])

    scrambled = SimpleNamespace(
        images=(
            SimpleNamespace(source_sequence_index=5, timestamp_ns=100),
            SimpleNamespace(source_sequence_index=2, timestamp_ns=200),
        )
    )
    with pytest.raises(ReconstructionValidationError, match="workspace did not preserve"):
        _validate_temporal_workspace_order(scrambled)

    non_monotonic_time = SimpleNamespace(
        images=(
            SimpleNamespace(source_sequence_index=2, timestamp_ns=200),
            SimpleNamespace(source_sequence_index=5, timestamp_ns=100),
        )
    )
    with pytest.raises(ReconstructionValidationError, match="timestamps"):
        _validate_temporal_workspace_order(non_monotonic_time)
