from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_gaussian_lifecycle_fault_env_uses_public_namespace() -> None:
    source = (
        ROOT / "scripts/gaussian_benchmark_workers/common.py"
    ).read_text(encoding="utf-8")

    assert (
        'LIFECYCLE_FAULT_ENV = '
        '"DEPTH3D_GAUSSIAN_LIFECYCLE_FAULT_INJECTION"'
        in source
    )
