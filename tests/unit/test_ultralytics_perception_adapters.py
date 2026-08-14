from __future__ import annotations

import sys
import types

import numpy as np

from core.contracts.common import FrameReference, ImageSize
from core.contracts.spatial import PixelCoordinateSpace, PixelSpaceRole
from core.perception.adapters import ultralytics_sam, ultralytics_yolo
from core.perception.adapters.ultralytics_sam import UltralyticsSAM2PromptedAdapter
from core.perception.adapters.ultralytics_yolo import UltralyticsYOLOSegmentationAdapter
from core.perception.messages import (
    FramePerceptionRequest,
    PromptedSegmentationRequest,
    SegmentationPrompt,
    SegmentationPromptKind,
)
from core.perception.metadata import PerceptionTask


def _request() -> FramePerceptionRequest:
    frame = FrameReference(1, 2, 3.0, ImageSize(4, 3))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    rgb[0, 0] = (1, 2, 3)
    return FramePerceptionRequest(
        frame,
        rgb,
        space,
        (PerceptionTask.DETECTION, PerceptionTask.INSTANCE_SEGMENTATION),
    )


def test_yolo_adapter_normalizes_results_and_keeps_tensors_out_of_contracts(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeBoxes:
        xyxy = np.asarray([[0.0, 0.0, 3.0, 2.0]], dtype=np.float32)
        conf = np.asarray([0.9], dtype=np.float32)
        cls = np.asarray([2.0], dtype=np.float32)

    class FakeMasks:
        data = np.asarray([[[1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]]], dtype=np.float32)

    class FakeResult:
        boxes = FakeBoxes()
        masks = FakeMasks()
        names = {2: "part"}

    class FakeYOLO:
        def __init__(self, checkpoint: str) -> None:
            captured["checkpoint"] = checkpoint

        def predict(self, **kwargs):
            captured.update(kwargs)
            return [FakeResult()]

    fake = types.ModuleType("ultralytics")
    fake.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    monkeypatch.setattr(ultralytics_yolo, "package_version", lambda _: "8.4-test")

    adapter = UltralyticsYOLOSegmentationAdapter()
    adapter.load({"checkpoint": "yolo11n-seg.pt", "device": "cpu", "retina_masks": True})
    batch = adapter.infer_frame(_request())

    assert captured["checkpoint"] == "yolo11n-seg.pt"
    assert np.asarray(captured["source"])[0, 0].tolist() == [3, 2, 1]
    assert captured["retina_masks"] is True
    assert batch.instances[0].class_name == "part"
    assert batch.instances[0].mask.dtype == np.bool_
    assert isinstance(batch.instances[0].box_xyxy, tuple)


def test_sam_adapter_routes_box_prompt_without_model_specific_worker_logic(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeMasks:
        data = np.ones((1, 3, 4), dtype=np.float32)

    class FakeResult:
        masks = FakeMasks()

    class FakeSAM:
        def __init__(self, checkpoint: str) -> None:
            captured["checkpoint"] = checkpoint

        def predict(self, **kwargs):
            captured.update(kwargs)
            return [FakeResult()]

    fake = types.ModuleType("ultralytics")
    fake.SAM = FakeSAM
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    monkeypatch.setattr(ultralytics_sam, "package_version", lambda _: "8.4-test")

    request = _request()
    prompt = SegmentationPrompt(
        "p",
        SegmentationPromptKind.BOX,
        request.rgb_space,
        box_xyxy=(0.0, 0.0, 3.0, 2.0),
    )
    adapter = UltralyticsSAM2PromptedAdapter()
    adapter.load({"checkpoint": "sam2.1_t.pt", "device": "cpu"})
    raw = adapter.refine_mask(
        PromptedSegmentationRequest(request, prompt, parent_detection_id="det")
    )

    assert captured["checkpoint"] == "sam2.1_t.pt"
    assert captured["bboxes"] == [0.0, 0.0, 3.0, 2.0]
    assert np.asarray(captured["source"])[0, 0].tolist() == [3, 2, 1]
    assert raw.mask.shape == (3, 4)


def test_sam_adapter_segment_everything_uses_no_prompt_and_returns_unlabeled_masks(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeMasks:
        # Deliberately use a smaller mask grid; the adapter must restore source size.
        data = np.asarray(
            [
                [[1, 1], [1, 0]],
                [[0, 1], [1, 1]],
            ],
            dtype=np.float32,
        )

    class FakeResult:
        masks = FakeMasks()

    class FakeSAM:
        def __init__(self, checkpoint: str) -> None:
            captured["checkpoint"] = checkpoint

        def predict(self, **kwargs):
            captured.update(kwargs)
            return [FakeResult()]

    fake = types.ModuleType("ultralytics")
    fake.SAM = FakeSAM
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    monkeypatch.setattr(ultralytics_sam, "package_version", lambda _: "8.4-test")

    request = _request()
    adapter = UltralyticsSAM2PromptedAdapter()
    adapter.load({"checkpoint": "sam2.1_t.pt", "device": "cpu", "imgsz": 1024})
    batch = adapter.propose_masks(request)

    assert captured["checkpoint"] == "sam2.1_t.pt"
    assert "bboxes" not in captured
    assert "points" not in captured
    assert "labels" not in captured
    assert np.asarray(captured["source"])[0, 0].tolist() == [3, 2, 1]
    assert len(batch.proposals) == 2
    assert all(proposal.mask.shape == (3, 4) for proposal in batch.proposals)
    assert batch.model_provenance.task == "automatic_segmentation"
