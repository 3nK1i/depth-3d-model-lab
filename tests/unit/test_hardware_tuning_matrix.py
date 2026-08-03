"""Deterministic cross-GPU tuning matrix used without owning every GPU."""

from __future__ import annotations

from scripts.check_hardware import _report, _simulated_hardware


def test_simulated_rtx2060_matrix() -> None:
    report = _report(_simulated_hardware("rtx2060"))
    profile = report["automatic_profile"]
    assert profile["effective"] == "efficient"
    assert profile["model_repo"] == "Ruicheng/moge-2-vits-normal"
    assert profile["precision"] == "fp16"
    assert profile["static_point_budget"] == 250_000
    assert profile["live_point_budget"] == 100_000
    assert profile["allow_tf32"] is False


def test_simulated_rtx5090_matrix() -> None:
    report = _report(_simulated_hardware("rtx5090"))
    profile = report["automatic_profile"]
    assert profile["effective"] == "ultra"
    assert profile["model_repo"] == "Ruicheng/moge-2-vitl-normal"
    assert profile["precision"] == "fp16"
    assert profile["static_point_budget"] == 1_000_000
    assert profile["live_point_budget"] == 500_000
    assert profile["warmup_runs"] == 2
    assert profile["allow_tf32"] is True


def test_simulated_5090_memory_pressure_is_safe() -> None:
    report = _report(_simulated_hardware("rtx5090-pressure"))
    profile = report["automatic_profile"]
    assert profile["effective"] == "balanced"
    assert profile["model_repo"] == "Ruicheng/moge-2-vitb-normal"


def test_release_gate_can_require_a_real_cuda_device(monkeypatch, capsys) -> None:
    import sys

    from scripts.check_hardware import main

    monkeypatch.setattr(
        sys,
        "argv",
        ["check_hardware.py", "--simulate", "cpu", "--require-cuda"],
    )

    assert main() == 4
    assert "CUDA requirement: FAIL" in capsys.readouterr().out
