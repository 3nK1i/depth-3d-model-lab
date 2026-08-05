"""Factories for Step 5 requests."""
from .fitting_pipeline import ObjectGeometryFitRequest
from .fitting_policy import ObjectGeometryFittingPolicy

def make_fitting_request(extraction_batch, *, observation_ids=(), policy=None):
    return ObjectGeometryFitRequest(extraction_batch=extraction_batch,observation_ids=tuple(observation_ids),policy=policy or ObjectGeometryFittingPolicy())
