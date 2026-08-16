from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from core.perception.semantic_project import SEGMENTATION_PROJECT_SCHEMA_VERSION
from core.perception.semantic_workspace_persistence import SEMANTIC_WORKSPACE_PERSISTENCE_SCHEMA_VERSION
from core.version import RELEASE_NAME, RELEASE_PHASE, __version__


ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "scripts" / "run_phase9_2i_closure_smoke.py"


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("phase9_2i_closure_smoke", SMOKE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_2i_closure_smoke_covers_the_phase92_semantic_end_state() -> None:
    module = _load_smoke_module()
    assert tuple(item[0] for item in module.SMOKES) == (
        "semantic_workspace",
        "segmentation_project",
        "auto_segment",
        "prompt_refinement",
        "deterministic_merge",
        "reviewed_measurement",
        "workspace_persistence",
    )


def test_2i_combined_closure_smoke_passes_and_keeps_semantics_non_authoritative() -> None:
    proc = subprocess.run(
        [sys.executable, str(SMOKE)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=240,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "PASS"
    assert all(payload["checks"].values())
    assert payload["ui_qualification"]["new_ui_in_9_2i"] is False
    assert payload["authority"] == {
        "semantic_geometry_authority": False,
        "semantic_metric_authority": False,
        "semantic_measurement_authority": False,
        "certified_metrology": False,
    }


def test_2i_closure_does_not_prematurely_change_release_or_semantic_schema_identity() -> None:
    assert __version__ == "0.9.0"
    assert RELEASE_NAME == "Phase 9 Digital Twin Stable"
    assert RELEASE_PHASE == 9.0
    assert SEGMENTATION_PROJECT_SCHEMA_VERSION == 1
    assert SEMANTIC_WORKSPACE_PERSISTENCE_SCHEMA_VERSION == 1
