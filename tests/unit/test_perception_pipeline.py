from __future__ import annotations
import numpy as np
from core.contracts.common import FrameReference, ImageSize, ModelProvenance
from core.contracts.spatial import PixelCoordinateSpace, PixelSpaceRole
from core.models.metadata import ConfidenceKind
from core.perception.adapters.mock_adapter import DevelopmentPerceptionMockAdapter
from core.perception.messages import FramePerceptionRequest
from core.perception.metadata import PerceptionTask
from core.perception.pipeline import PerceptionPipeline
from core.perception.registry import PERCEPTION_MODEL_REGISTRY
from core.perception.result import ApprovalStatus, DenseMaskPayload, InstanceMask, MaskSource

def request():
    f = FrameReference(3, 4, 1.0, ImageSize(20, 16))
    s = PixelCoordinateSpace('rgb', f, f.source_size, PixelSpaceRole.RGB)
    return FramePerceptionRequest(f, np.zeros((16, 20, 3), np.uint8), s, (PerceptionTask.DETECTION, PerceptionTask.INSTANCE_SEGMENTATION))

def adapter(native=True):
    a = DevelopmentPerceptionMockAdapter()
    a.load({'emit_native_masks': native})
    a.warmup()
    return a

def test_native_segmentation_multiple_objects_and_overlaps():
    a = adapter()
    r = PerceptionPipeline(a, PERCEPTION_MODEL_REGISTRY.get('development-perception-mock')).process_frame(request())
    assert len(r.observations) == 2
    assert all((o.selected_mask.source is MaskSource.NATIVE_INSTANCE for o in r.observations))
    assert np.any(r.observations[0].selected_mask.payload.data & r.observations[1].selected_mask.payload.data)

def test_prompted_refinement_precedes_box_fallback():
    a = adapter(False)
    ref = adapter()
    r = PerceptionPipeline(a, PERCEPTION_MODEL_REGISTRY.get('development-perception-mock'), refiner=ref, refiner_descriptor=PERCEPTION_MODEL_REGISTRY.get('development-perception-mock')).process_frame(request())
    assert all((o.selected_mask.source is MaskSource.PROMPTED_REFINEMENT for o in r.observations))
    assert all((len(o.refinements) == 1 for o in r.observations))
    assert all((o.refinements[0].output_mask.mask_id == o.selected_mask_id for o in r.observations))

def test_detection_box_fallback_when_refiner_absent():
    a = adapter(False)
    r = PerceptionPipeline(a, PERCEPTION_MODEL_REGISTRY.get('development-perception-mock')).process_frame(request())
    assert all((o.selected_mask.source is MaskSource.DETECTION_BOX_FALLBACK for o in r.observations))

def test_approved_manual_mask_has_absolute_precedence():
    req = request()
    a = adapter()
    parent = np.ones((16, 20), bool)
    manual = InstanceMask('manual', req.frame_reference, req.rgb_space, DenseMaskPayload(parent), MaskSource.MANUAL_CORRECTION, ModelProvenance('user', 'Manual', 'manual'), revision=1, parent_mask_id='old', approval_status=ApprovalStatus.APPROVED, approved_by='reviewer', approved_timestamp=1.0, foreground_valid=True)
    r = PerceptionPipeline(a, PERCEPTION_MODEL_REGISTRY.get('development-perception-mock')).process_frame(req, manual_masks={'mock-0': (manual,)})
    assert r.observations[0].selected_mask_id == 'manual'

def test_pending_manual_mask_does_not_override_automatic_refinement():
    req = request()
    a = adapter(False)
    ref = adapter()
    pending = InstanceMask('pending', req.frame_reference, req.rgb_space, DenseMaskPayload(np.ones((16, 20), bool)), MaskSource.MANUAL_CORRECTION, ModelProvenance('user', 'Manual', 'manual'), revision=1, parent_mask_id='old', approval_status=ApprovalStatus.PENDING, foreground_valid=True)
    r = PerceptionPipeline(a, PERCEPTION_MODEL_REGISTRY.get('development-perception-mock'), refiner=ref, refiner_descriptor=PERCEPTION_MODEL_REGISTRY.get('development-perception-mock')).process_frame(req, manual_masks={'mock-0': (pending,)})
    assert r.observations[0].selected_mask.source is MaskSource.PROMPTED_REFINEMENT
    assert any((mask.mask_id == 'pending' for mask in r.observations[0].masks))
