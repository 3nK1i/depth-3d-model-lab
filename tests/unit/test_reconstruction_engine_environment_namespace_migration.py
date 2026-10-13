from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

ENGINE = ROOT / "core/reconstruction/engine_hardening.py"
FAULT = ROOT / "scripts/reconstruction_engine_fault_worker.py"
FROZEN = ROOT / "scripts/i5_11_bounded_fault_worker.py"
INTEGRATED = ROOT / "scripts/reconstruction_engine_integrated_worker.py"

FROZEN_SHA256 = "5d3557e1c07b87816800008de0569e1f86d388657f2fbedf1200b8518c58fae7"


def test_engine_workspace_producer_emits_public_and_frozen_compatibility_contract() -> None:
    source = ENGINE.read_text(encoding="utf-8")

    assert 'env["DEPTH3D_RECONSTRUCTION_ENGINE_WORKSPACE"] = str(workspace)' in source
    assert 'env["EIA_RECONSTRUCTION_ENGINE_WORKSPACE"] = str(workspace)' in source


def test_non_frozen_engine_workers_use_public_workspace_namespace() -> None:
    for path in (FAULT, INTEGRATED):
        source = path.read_text(encoding="utf-8")

        assert 'os.environ.get("DEPTH3D_RECONSTRUCTION_ENGINE_WORKSPACE", "")' in source
        assert "EIA_RECONSTRUCTION_ENGINE_WORKSPACE" not in source


def test_i511_bounded_fault_worker_remains_byte_frozen_on_legacy_contract() -> None:
    source = FROZEN.read_text(encoding="utf-8")

    assert 'os.environ.get("EIA_RECONSTRUCTION_ENGINE_WORKSPACE", "")' in source
    assert "DEPTH3D_RECONSTRUCTION_ENGINE_WORKSPACE" not in source

    digest = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    assert digest == FROZEN_SHA256
