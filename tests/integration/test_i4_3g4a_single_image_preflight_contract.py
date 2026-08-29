from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from core.reconstruction.single_image_provider_qualification import (
    I43G4A_G2_SCENARIO_MAP,
    I43G4A_PROVIDER_IMPLEMENTATIONS,
    I43G4A_SCENARIOS,
)


def test_g4a_cli_exposes_explicit_predecessor_source_runtime_storage_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/run_i4_3g4a_single_image_preflight.py"), "--help"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    output = completed.stdout
    for flag in (
        "--i4-3g1-snapshot",
        "--i4-3g3c-closure",
        "--complete-room-g2b-snapshot",
        "--low-texture-g2b-snapshot",
        "--weak-overlap-g2b-snapshot",
        "--minimum-free-gib",
        "--output-dir",
    ):
        assert flag in output
    for implementation in I43G4A_PROVIDER_IMPLEMENTATIONS:
        assert f"--{implementation.replace('_', '-')}-runtime-json" in output


def test_g4a_frozen_scenario_reuse_map_is_one_to_one_and_complete() -> None:
    assert tuple(I43G4A_G2_SCENARIO_MAP) == I43G4A_SCENARIOS
    assert len(set(I43G4A_G2_SCENARIO_MAP.values())) == 3
    assert I43G4A_G2_SCENARIO_MAP["g4_single_image_complete_room"] == "g2_complete_room_mvs_da3"
    assert I43G4A_G2_SCENARIO_MAP["g4_single_image_low_texture"] == "g2_low_texture_mvs_da3"
    assert I43G4A_G2_SCENARIO_MAP["g4_single_image_weak_overlap"] == "g2_weak_overlap_mvs_da3"
