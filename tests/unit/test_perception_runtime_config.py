from core.hardware.gpu_info import AcceleratorBackend, HardwareInfo
from core.perception.registry import PERCEPTION_MODEL_REGISTRY
from core.perception.runtime_config import PerceptionComputeProfile, recommended_perception_profile, resolve_perception_runtime

def hw(gib, cuda=True):
    return HardwareInfo(AcceleratorBackend.CUDA if cuda else AcceleratorBackend.CPU, 'cuda:0' if cuda else 'cpu', 0 if cuda else None, 'GPU' if cuda else 'CPU', gib * 1024 ** 3 if cuda else None, gib * 1024 ** 3 if cuda else None, (7, 5) if cuda else None, cuda, False, True, cuda, 'x', 'x' if cuda else None, ('sm_75',) if cuda else (), True if cuda else None)

def test_2060_uses_efficient_yolo11n_seg():
    r = resolve_perception_runtime(PERCEPTION_MODEL_REGISTRY.get('ultralytics-yolo-seg'), hw(6))
    assert r.effective_profile is PerceptionComputeProfile.EFFICIENT
    assert r.checkpoint == 'yolo11n-seg.pt'
    assert r.precision == 'fp16'

def test_5090_uses_ultra_but_starts_yolo11s():
    r = resolve_perception_runtime(PERCEPTION_MODEL_REGISTRY.get('ultralytics-yolo-seg'), hw(32))
    assert r.effective_profile is PerceptionComputeProfile.ULTRA
    assert r.checkpoint == 'yolo11s-seg.pt'
    assert r.imgsz == 960

def test_sam_profile_uses_sam21_weights_and_cpu_fp32():
    r = resolve_perception_runtime(PERCEPTION_MODEL_REGISTRY.get('ultralytics-sam2-prompted'), hw(0, False))
    assert r.checkpoint == 'sam2.1_t.pt'
    assert r.device == 'cpu'
    assert r.precision == 'fp32'
