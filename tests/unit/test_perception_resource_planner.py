from core.hardware.gpu_info import AcceleratorBackend, HardwareInfo
from core.perception.registry import PERCEPTION_MODEL_REGISTRY
from core.perception.resource_planner import PerceptionExecutionMode, plan_perception_resources

def hw(free=32):
    return HardwareInfo(AcceleratorBackend.CUDA, 'cuda:0', 0, 'GPU', 32 * 1024 ** 3, free * 1024 ** 3, (12, 0), True, True, True, True, 'x', 'x', ('sm_120',), True)

def test_live_unknown_vram_disables_refinement():
    p = plan_perception_resources(PERCEPTION_MODEL_REGISTRY.get('ultralytics-yolo-seg'), PERCEPTION_MODEL_REGISTRY.get('ultralytics-sam2-prompted'), hw(), live=True)
    assert p.mode is PerceptionExecutionMode.PRIMARY_ONLY

def test_offline_unknown_vram_uses_sequential():
    p = plan_perception_resources(PERCEPTION_MODEL_REGISTRY.get('ultralytics-yolo-detect'), PERCEPTION_MODEL_REGISTRY.get('ultralytics-sam2-prompted'), hw(), live=False)
    assert p.mode is PerceptionExecutionMode.SEQUENTIAL_OFFLINE

def test_measured_5090_co_residency_needs_margin():
    p = plan_perception_resources(PERCEPTION_MODEL_REGISTRY.get('ultralytics-yolo-seg'), PERCEPTION_MODEL_REGISTRY.get('ultralytics-sam2-prompted'), hw(), live=True, measured_primary_vram_mb=2000, measured_refiner_vram_mb=6000)
    assert p.mode is PerceptionExecutionMode.CO_RESIDENT
    assert p.safety_margin_mb > 0
