from pathlib import Path
import importlib.util

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_real_measurement_validation.py"


def _module():
    spec = importlib.util.spec_from_file_location("real_measurement_validation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _real_chain():
    from core.measurement import compute_measurements, make_measurement_request
    from tests.measurement_computation_helpers import make_chain

    bundle, extraction, fitting = make_chain()
    measurements = compute_measurements(
        make_measurement_request(bundle, extraction, fitting)
    )
    return bundle, extraction, fitting, measurements


def test_parser_defaults_are_safe_and_sequential():
    module = _module()
    parser = module.build_parser()
    args = parser.parse_args(["--image", "input.png"])
    assert args.depth_model == "depth-pro"
    assert args.perception_model == "ultralytics-yolo-seg"
    assert args.depth_profile == "efficient"
    assert args.perception_profile == "efficient"
    assert args.max_objects == 5


def test_depth_preview_is_bgr_uint8():
    module = _module()
    result = module._depth_preview(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), None
    )
    assert result.shape == (2, 2, 3)
    assert result.dtype == np.uint8


def test_color_palette_is_deterministic():
    module = _module()
    assert module._color_for(0) == module._color_for(5)


def test_measurement_lines_use_authoritative_estimates_contract():
    module = _module()
    _, _, _, batch = _real_chain()
    computation = batch.object_results[0]

    lines = module._measurement_lines(computation)

    assert len(lines) == len(computation.measurement_result.estimates)
    assert any(line.startswith("width:") for line in lines)
    assert any("+/-" in line for line in lines)


def test_serialization_includes_extraction_fitting_boxes_and_measurements():
    module = _module()
    _, extraction, fitting, measurements = _real_chain()

    payload = module._serialize(
        measurements,
        extraction,
        fitting,
        {"depth_model": "test"},
    )

    expected = measurements.object_results[0].measurement_result.estimates
    item = payload["objects"][0]
    assert len(payload["objects"]) == 1
    assert item["extraction"]["status"] == "success"
    assert item["extraction"]["stage_counts"]["output_points"] > 0
    assert item["fitting"]["preferred_box_id"]
    assert item["fitting"]["boxes"]
    assert len(item["measurements"]) == len(expected)
    assert item["measurements"][0]["kind"] == expected[0].kind.value


def test_box_corner_generation_and_projection_are_finite():
    module = _module()
    bundle, _, fitting, _ = _real_chain()
    fit = fitting.object_results[0]
    assert fit.oriented_box is not None

    corners = module._box_corners(fit.oriented_box)
    uv, valid = module._project_camera_points(corners, bundle.intrinsics.matrix)

    assert corners.shape == (8, 3)
    assert uv.shape == (8, 2)
    assert valid.all()
    assert np.isfinite(uv).all()


def test_points_panel_draws_fitted_boxes():
    module = _module()
    _, extraction, fitting, _ = _real_chain()
    panel = module._points_panel(
        extraction.object_results[0], fitting.object_results[0], 600, 320
    )
    assert panel.shape == (320, 600, 3)
    assert panel.dtype == np.uint8
    assert np.count_nonzero(panel) > 0

def test_xz_box_diagnostic_uses_convex_silhouette_without_crossing_edges():
    module = _module()
    _, _, fitting, _ = _real_chain()
    box = fitting.object_results[0].oriented_box
    assert box is not None

    hull = module._box_xz_hull(box)

    assert hull.ndim == 2
    assert hull.shape[1] == 2
    assert 3 <= len(hull) <= 8
    assert np.isfinite(hull).all()


def test_serialization_includes_stable_object_index():
    module = _module()
    _, extraction, fitting, measurements = _real_chain()

    payload = module._serialize(measurements, extraction, fitting, {"depth_model": "test"})

    assert payload["objects"][0]["object_index"] == 0
