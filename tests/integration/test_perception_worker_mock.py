from __future__ import annotations
import multiprocessing as mp
import numpy as np
from core.contracts.common import FrameReference, ImageSize
from core.contracts.spatial import PixelCoordinateSpace, PixelSpaceRole
from core.perception.messages import FramePerceptionRequest, PerceptionPipelineRequest, PerceptionShutdownRequest, PerceptionWorkerConfiguration, PerceptionWorkerReady, PerceptionWorkerResponse
from core.perception.metadata import PerceptionTask
from core.perception.worker import perception_worker_main

def test_mock_worker_round_trip():
    ctx = mp.get_context('spawn')
    iq = ctx.Queue(1)
    oq = ctx.Queue(8)
    p = ctx.Process(target=perception_worker_main, args=(PerceptionWorkerConfiguration('development-perception-mock'), iq, oq))
    p.start()
    assert isinstance(oq.get(timeout=10), PerceptionWorkerReady)
    f = FrameReference(1, 1, 0, ImageSize(8, 6))
    s = PixelCoordinateSpace('rgb', f, f.source_size, PixelSpaceRole.RGB)
    iq.put(PerceptionPipelineRequest(7, FramePerceptionRequest(f, np.zeros((6, 8, 3), np.uint8), s, (PerceptionTask.DETECTION,))))
    response = oq.get(timeout=10)
    assert isinstance(response, PerceptionWorkerResponse)
    assert response.error == ''
    assert len(response.result.observations) == 2
    iq.put(PerceptionShutdownRequest())
    p.join(10)
    assert not p.is_alive()
