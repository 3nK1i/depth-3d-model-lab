from __future__ import annotations
import pytest
from core.perception.base import BasePerceptionAdapter,FramePerceptionAdapter
from core.perception.metadata import AdapterCapability,PerceptionCapabilities,PerceptionDescriptor,PerceptionTask
from core.perception.registry import PERCEPTION_MODEL_REGISTRY,PerceptionAdapterContractError,PerceptionRegistry
class FrameOnlyAdapter(BasePerceptionAdapter):
 def load(self,config):pass
 def warmup(self):pass
 def unload(self):pass
 def get_metadata(self):return {}
 def infer_frame(self,request):return request
class LifecycleOnlyAdapter(BasePerceptionAdapter):
 def load(self,config):pass
 def warmup(self):pass
 def unload(self):pass
 def get_metadata(self):return {}
def descriptor(name):return PerceptionDescriptor('test-perception','Test',f'tests.unit.test_perception_registry:{name}',PerceptionCapabilities((PerceptionTask.DETECTION,),(AdapterCapability.FRAME_PERCEPTION,)))
def test_production_registry_contains_approved_step2_descriptors():assert PERCEPTION_MODEL_REGISTRY.ids()==('ultralytics-yolo-seg','ultralytics-yolo-detect','ultralytics-sam2-prompted','development-perception-mock')
def test_registry_builds_declared_protocol():
 r=PerceptionRegistry();r.register(descriptor('FrameOnlyAdapter'));assert isinstance(r.build('test-perception'),FramePerceptionAdapter)
def test_registry_rejects_missing_protocol():
 r=PerceptionRegistry();r.register(descriptor('LifecycleOnlyAdapter'))
 with pytest.raises(PerceptionAdapterContractError,match='infer_frame'):r.build('test-perception')
