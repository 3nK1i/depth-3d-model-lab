from __future__ import annotations

from core.reconstruction import ReconstructionTrajectory
from tests.reconstruction_helpers import make_pose


def test_large_ordered_trajectory_contract_remains_deterministic() -> None:
    samples = tuple(make_pose(index) for index in range(2_000))
    trajectory = ReconstructionTrajectory(samples)
    assert len(trajectory.samples) == 2_000
    assert trajectory.samples[-1].timestamp_ns == 2_000_000_000
