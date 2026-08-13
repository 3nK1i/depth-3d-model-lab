from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from core.reconstruction.persistent_scene import PERSISTENT_RECONSTRUCTION_SCENE_SCHEMA_VERSION
from core.version import RELEASE_NAME, RELEASE_PHASE, __version__


ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "scripts" / "run_phase9_1i_closure_smoke.py"


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("phase9_1i_closure_smoke", SMOKE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_1i_closure_smoke_covers_the_four_phase91_end_state_authorities() -> None:
    module = _load_smoke_module()
    assert module.SMOKES == (
        ("automatic_calibration", "scripts/run_phase9_1d_automatic_calibration_smoke.py"),
        ("metric_qualification", "scripts/run_phase9_1e_metric_qualification_smoke.py"),
        ("measurement_calibration", "scripts/run_phase9_1g_measurement_calibration_smoke.py"),
        ("authority_persistence", "scripts/run_phase9_1h_persistence_smoke.py"),
    )


def test_1i_combined_closure_smoke_passes_and_keeps_deferred_field_validation_truthful() -> None:
    proc = subprocess.run(
        [sys.executable, str(SMOKE)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "PASS"
    assert all(payload["checks"].values())
    assert payload["certified_metrology"] is False
    assert payload["deferred_real_validation"] == {
        "physical_camera_calibration_and_measure_inspect": True,
        "real_persistent_scene_save_restart_reload": True,
        "reason": "No suitable real calibration/profile and persistent scene are currently available.",
    }


def test_1i_phase91_closure_does_not_prematurely_change_release_identity_or_scene_schema() -> None:
    assert __version__ == "0.8.0"
    assert RELEASE_NAME == "Phase 8 Reconstruction Stable"
    assert RELEASE_PHASE == 8.0
    assert PERSISTENT_RECONSTRUCTION_SCENE_SCHEMA_VERSION == 1
