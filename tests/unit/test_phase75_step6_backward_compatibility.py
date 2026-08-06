from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

from core.measurement.contracts import MeasurementEstimate, MeasurementKind, RejectionReason
from core.models.registry import MODEL_REGISTRY
from core.version import RELEASE_NAME, __version__


def test_step6_preserves_release_and_depth_registry_order() -> None:
    assert __version__ == "0.7.5"
    assert RELEASE_NAME == "Phase 7.5 Stable"
    assert MODEL_REGISTRY.ids() == (
        "moge-2",
        "depth-pro",
        "unidepth-v2",
        "metric3d-v2",
        "depth-anything-v2",
        "video-depth-anything",
        "marigold",
        "development-mock",
    )


def test_measurement_estimate_constructor_remains_append_compatible() -> None:
    names = list(inspect.signature(MeasurementEstimate).parameters)
    assert names[:10] == [
        "measurement_id",
        "observation_id",
        "kind",
        "status",
        "method",
        "unit_si",
        "reliability",
        "depth_scale",
        "intrinsics_source",
        "calibration_status",
    ]
    assert MeasurementKind.WIDTH.value == "width"
    assert RejectionReason.FITTING_FAILED.value == "shape_or_box_fitting_failed"


def test_step6_imports_do_not_load_optional_runtime_frameworks() -> None:
    """Check import isolation in a fresh interpreter.

    The full pytest suite legitimately imports Qt and PyTorch in other tests. Inspecting
    the parent process's global ``sys.modules`` therefore produces order-dependent false
    failures. This subprocess verifies the actual Step 6 requirement: importing the
    measurement pipeline must not *newly* import optional runtime frameworks.
    """

    repository_root = Path(__file__).resolve().parents[2]
    script = r'''
import importlib
import sys

optional_roots = ("torch", "ultralytics", "PyQt6", "vispy")
before = set(sys.modules)
importlib.import_module("core.measurement.pipeline")
new_modules = set(sys.modules) - before
unexpected = sorted(
    module_name
    for module_name in new_modules
    if any(
        module_name == root or module_name.startswith(root + ".")
        for root in optional_roots
    )
)
if unexpected:
    print("Unexpected optional imports:", ", ".join(unexpected))
    raise SystemExit(1)
'''

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        "Importing core.measurement.pipeline loaded optional runtime frameworks "
        f"in a fresh interpreter.\nstdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
