from pathlib import Path

from core.reconstruction import (
    DA3_PROVIDER_ID,
    PIPELINE_DENSE_PROVIDER_ID,
    PIPELINE_INSPECTION_PROVIDER_ID,
    PIPELINE_SEMANTICS_PROVIDER_ID,
    PIPELINE_SURFACE_PROVIDER_ID,
    ReconstructionPipelineStage,
    required_pipeline_stages,
)
from core.reconstruction.orchestration import ReconstructionPipelineGoal


def test_f1_stage_dag_is_not_reordered_by_f3():
    assert required_pipeline_stages((ReconstructionPipelineGoal.SEMANTIC_INSPECTION,)) == (
        ReconstructionPipelineStage.SOURCE_INGESTION,
        ReconstructionPipelineStage.CAMERA_POSE,
        ReconstructionPipelineStage.DEPTH_EVIDENCE,
        ReconstructionPipelineStage.DENSE_GEOMETRY,
        ReconstructionPipelineStage.SURFACE_MESH,
        ReconstructionPipelineStage.SEMANTIC_GEOMETRY,
        ReconstructionPipelineStage.INSPECTION_QC,
    )


def test_f2_da3_provider_identity_remains_supporting_provider():
    assert DA3_PROVIDER_ID == "depth-anything-3-external"
    assert PIPELINE_DENSE_PROVIDER_ID != DA3_PROVIDER_ID


def test_f3_provider_ids_reuse_existing_authority_boundaries():
    assert PIPELINE_DENSE_PROVIDER_ID == "canonical-dense-fusion"
    assert PIPELINE_SURFACE_PROVIDER_ID == "quality-aware-surface-mesh"
    assert PIPELINE_SEMANTICS_PROVIDER_ID == "semantic-geometry-core"
    assert PIPELINE_INSPECTION_PROVIDER_ID == "registration-inspection-qc"


def test_f3_core_has_no_heavy_da3_or_neural_runtime_imports():
    text = (Path(__file__).resolve().parents[2] / "core/reconstruction/pipeline_geometry.py").read_text()
    forbidden = (
        "import torch",
        "from torch",
        "import xformers",
        "from xformers",
        "import depth_anything_3",
        "from depth_anything_3",
        "import nerfstudio",
        "from nerfstudio",
        "import gsplat",
        "from gsplat",
        "import tinycudann",
        "from tinycudann",
    )
    assert not any(token in text for token in forbidden)


def test_f3_does_not_enable_ui_or_persistent_scene_early():
    text = (Path(__file__).resolve().parents[2] / "core/reconstruction/pipeline_geometry.py").read_text()
    assert 'reconstruction_ui_enabled: bool = False' in text
    assert 'persistent_reconstruction_scene_deferred_to' in text
