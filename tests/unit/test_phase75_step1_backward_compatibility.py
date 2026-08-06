from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from core.models.registry import MODEL_REGISTRY
from core.models.result import GeometryResult
from core.perception.registry import PERCEPTION_MODEL_REGISTRY
from core.runtime.messages import InferenceResult
from core.version import RELEASE_NAME, __version__


EXPECTED_DEPTH_MODEL_IDS = (
    "moge-2",
    "depth-pro",
    "unidepth-v2",
    "metric3d-v2",
    "depth-anything-v2",
    "video-depth-anything",
    "marigold",
    "development-mock",
)


def test_existing_depth_registry_order_is_unchanged_and_perception_is_separate() -> None:
    assert MODEL_REGISTRY.ids() == EXPECTED_DEPTH_MODEL_IDS
    assert PERCEPTION_MODEL_REGISTRY.ids() == (
        "ultralytics-yolo-seg",
        "ultralytics-yolo-detect",
        "ultralytics-sam2-prompted",
        "development-perception-mock",
    )


def test_geometry_result_phase5_positional_contract_is_unchanged() -> None:
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.ones((2, 3), dtype=np.float32)
    result = GeometryResult(
        1, 2, 3.0, rgb, depth, None, None, None, None, None, None, 1,
        4.0, 5.0, 6.0, 7.0, 8.0, True, "legacy", "Legacy", "FP32",
        0.1, 0.2, 0.3, 0.4, 0.5, 6.0,
    )
    assert result.model_id == "legacy"
    assert result.depth_scale == ""


def test_inference_result_ipc_contract_remains_pickle_compatible() -> None:
    result = InferenceResult(1, 2, "development-mock", None, 3.0)
    restored = pickle.loads(pickle.dumps(result))
    assert restored == result
    assert restored.confidence_kind == ""


def test_phase7_modes_and_phase75_release_identity_are_preserved() -> None:
    project_root = Path(__file__).resolve().parents[2]
    mode_source = (project_root / "app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    assert "_ENABLED_MODES = {AppMode.LIVE_3D, AppMode.INSPECTION, AppMode.STABLE_VIDEO, AppMode.SNAPSHOT, AppMode.BENCHMARK}" in mode_source
    assert "AppMode.RECONSTRUCTION" not in mode_source.split("_ENABLED_MODES =", 1)[1].split("\n", 1)[0]
    assert __version__ == "0.7.5"
    assert RELEASE_NAME == "Phase 7.5 Stable"


def test_step2_keeps_optional_runtime_imports_adapter_local() -> None:
    project_root = Path(__file__).resolve().parents[2]
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.7.5"' in pyproject
    source = "\n".join(path.read_text(encoding="utf-8") for path in sorted((project_root / "core/perception").glob("*.py")))
    assert "from ultralytics" not in source
    assert "import ultralytics" not in source
    assert "PyQt" not in source
