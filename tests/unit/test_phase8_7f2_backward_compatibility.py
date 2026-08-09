from pathlib import Path
from core.reconstruction import (
    RECONSTRUCTION_ORCHESTRATION_SCHEMA_VERSION, NERF_E4_SCHEMA_VERSION,
    GAUSSIAN_SPLATTING_SCHEMA_VERSION, DA3_SCHEMA_VERSION,
)
ROOT=Path(__file__).resolve().parents[2]

def test_schema_versions_stay_stable():
    assert RECONSTRUCTION_ORCHESTRATION_SCHEMA_VERSION==1
    assert NERF_E4_SCHEMA_VERSION==1
    assert GAUSSIAN_SPLATTING_SCHEMA_VERSION==1
    assert DA3_SCHEMA_VERSION==1

def test_production_pyproject_does_not_gain_da3_dependencies():
    text=(ROOT/"pyproject.toml").read_text().lower()
    for token in ("depth-anything-3","depth_anything_3","xformers"):
        assert token not in text

def test_core_never_imports_da3_torch_or_xformers_directly():
    for name in ("da3.py","da3_provider.py"):
        text=(ROOT/"core/reconstruction"/name).read_text().lower()
        forbidden=("import torch","from torch","import xformers","from xformers","import depth_anything_3","from depth_anything_3")
        assert not any(token in text for token in forbidden)

def test_f2_does_not_bind_canonical_geometry_early():
    text=(ROOT/"core/reconstruction/da3_provider.py").read_text()
    assert "ReconstructionProviderRole.DENSE_GEOMETRY" not in text
    assert "ReconstructionProviderRole.SEMANTICS" not in text
    assert "ReconstructionProviderRole.INSPECTION_QC" not in text

def test_f2_does_not_implement_persistent_scene():
    for p in (ROOT/"core/reconstruction").glob("da3*.py"):
        assert "class PersistentReconstructionScene" not in p.read_text()

def test_ui_remains_disabled():
    assert "reconstruction_mode_enabled: true" not in (ROOT/"config/app.yaml").read_text().lower()

def test_inference_controller_remains_uncoupled():
    path=ROOT/"app/inference_controller.py"; text=path.read_text().lower()
    assert "depthanything3providerexecutor" not in text and "core.reconstruction.da3" not in text
