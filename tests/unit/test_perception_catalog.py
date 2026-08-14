from core.perception.metadata import AdapterCapability, CoResidencyPolicy, PerceptionTask
from core.perception.registry import PERCEPTION_MODEL_REGISTRY
EXPECTED = ('ultralytics-yolo-seg', 'ultralytics-yolo-detect', 'ultralytics-sam2-prompted', 'development-perception-mock')

def test_builtin_registry_order_and_separation():
    assert PERCEPTION_MODEL_REGISTRY.ids() == EXPECTED

def test_yolo_and_sam_capabilities_are_separate():
    seg = PERCEPTION_MODEL_REGISTRY.get('ultralytics-yolo-seg')
    sam = PERCEPTION_MODEL_REGISTRY.get('ultralytics-sam2-prompted')
    assert PerceptionTask.INSTANCE_SEGMENTATION in seg.capabilities.tasks
    assert AdapterCapability.PROMPTED_SEGMENTATION not in seg.capabilities.adapter_capabilities
    assert AdapterCapability.PROMPTED_SEGMENTATION in sam.capabilities.adapter_capabilities
    assert AdapterCapability.AUTOMATIC_SEGMENTATION in sam.capabilities.adapter_capabilities
    assert PerceptionTask.AUTOMATIC_SEGMENTATION in sam.capabilities.tasks

def test_unmeasured_production_resources_prefer_isolation():
    for mid in EXPECTED[:3]:
        r = PERCEPTION_MODEL_REGISTRY.get(mid).resources
        assert r.estimated_vram_mb is None
        assert r.co_residency_policy is CoResidencyPolicy.PREFER_ISOLATED
