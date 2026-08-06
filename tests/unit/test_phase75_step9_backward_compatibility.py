from core.inspection import load_inspection_bundle
from core.measurement.pipeline import compute_measurements


def test_step9_keeps_step8_bundle_reader_and_measurement_pipeline() -> None:
    assert callable(load_inspection_bundle)
    assert callable(compute_measurements)
