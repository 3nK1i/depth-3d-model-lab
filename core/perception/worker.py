"""Generic perception worker; routes by typed request and capability only."""
from __future__ import annotations
import queue
from core.perception.base import SequencePerceptionAdapter
from core.perception.messages import PerceptionPipelineRequest, PerceptionShutdownRequest, PerceptionWorkerConfiguration, PerceptionWorkerReady, PerceptionWorkerResponse, PerceptionWorkerStopped, SequenceResetRequest
from core.perception.pipeline import PerceptionPipeline, PerceptionPipelinePolicy
from core.perception.registry import PERCEPTION_MODEL_REGISTRY

def perception_worker_main(configuration: PerceptionWorkerConfiguration, input_queue, output_queue) -> None:
    primary = None
    refiner = None
    try:
        primary = PERCEPTION_MODEL_REGISTRY.build(configuration.primary_model_id)
        primary.load(dict(configuration.primary_config))
        primary.warmup()
        primary_descriptor = PERCEPTION_MODEL_REGISTRY.get(configuration.primary_model_id)
        refiner_descriptor = None
        if configuration.refiner_model_id:
            if not configuration.allow_refiner_co_residency:
                raise RuntimeError('Refiner co-residency was not explicitly approved by the resource planner.')
            refiner = PERCEPTION_MODEL_REGISTRY.build(configuration.refiner_model_id)
            refiner.load(dict(configuration.refiner_config))
            refiner.warmup()
            refiner_descriptor = PERCEPTION_MODEL_REGISTRY.get(configuration.refiner_model_id)
        pipeline = PerceptionPipeline(primary, primary_descriptor, refiner=refiner, refiner_descriptor=refiner_descriptor, policy=PerceptionPipelinePolicy(configuration.enable_prompted_refinement, configuration.allow_box_fallback, configuration.minimum_mask_pixels))
        output_queue.put(PerceptionWorkerReady(configuration.primary_model_id, configuration.refiner_model_id))
        while True:
            message = input_queue.get()
            if isinstance(message, PerceptionShutdownRequest):
                break
            if isinstance(message, SequenceResetRequest):
                for adapter in (primary, refiner):
                    if adapter is not None and isinstance(adapter, SequencePerceptionAdapter):
                        adapter.reset_sequence(message)
                continue
            if isinstance(message, PerceptionPipelineRequest):
                try:
                    output_queue.put(PerceptionWorkerResponse(message.request_id, pipeline.process_frame(message.frame_request, manual_masks=dict(message.manual_masks))))
                except Exception as exc:
                    output_queue.put(PerceptionWorkerResponse(message.request_id, None, f'{type(exc).__name__}: {exc}', False))
    except Exception as exc:
        output_queue.put(PerceptionWorkerResponse(-1, None, f'{type(exc).__name__}: {exc}', True))
    finally:
        for adapter in (refiner, primary):
            if adapter is not None:
                try:
                    adapter.unload()
                except Exception:
                    pass
        try:
            output_queue.put(PerceptionWorkerStopped())
        except Exception:
            pass
