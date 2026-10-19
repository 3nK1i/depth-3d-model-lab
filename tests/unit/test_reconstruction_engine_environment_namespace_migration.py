from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

ENGINE = ROOT / "core/reconstruction/engine_hardening.py"
FAULT = ROOT / "scripts/reconstruction_engine_fault_worker.py"
FROZEN = ROOT / "scripts/i5_11_bounded_fault_worker.py"
INTEGRATED = ROOT / "scripts/reconstruction_engine_integrated_worker.py"

FROZEN_SHA256 = "3a94eb740230f9b2d9d2698bae1170883c0a8b4d662521dfb9d783dcc7c13836"


def test_engine_workspace_producer_emits_public_contract() -> None:
    source = ENGINE.read_text(encoding="utf-8")

    assert (
        source.count(
            'env["DEPTH3D_RECONSTRUCTION_ENGINE_WORKSPACE"] = str(workspace)'
        )
        == 1
    )


def test_all_engine_workers_use_public_workspace_namespace() -> None:
    for path in (FAULT, INTEGRATED, FROZEN):
        source = path.read_text(encoding="utf-8")

        assert (
            'os.environ.get("DEPTH3D_RECONSTRUCTION_ENGINE_WORKSPACE", "")'
            in source
        )


def test_i511_bounded_fault_worker_remains_frozen_at_public_contract() -> None:
    digest = hashlib.sha256(
        FROZEN.read_bytes()
    ).hexdigest()

    assert digest == FROZEN_SHA256
