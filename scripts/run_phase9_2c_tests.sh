#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"
case "$MODE" in
  quick)
    python -m pytest -q \
      tests/unit/test_phase9_2c_auto_segment_engine.py \
      tests/unit/test_ultralytics_perception_adapters.py \
      tests/unit/test_perception_catalog.py \
      tests/unit/test_perception_metadata.py \
      tests/unit/test_perception_registry.py \
      tests/unit/test_phase9_2b_semantic_project.py \
      tests/unit/test_phase9_2a_semantic_workspace.py
    ;;
  integration)
    python -m pytest -q \
      tests/integration/test_phase9_2c_auto_segment_project_flow.py \
      tests/integration/test_phase9_2b_semantic_project_workspace.py \
      tests/integration/test_phase9_2a_semantic_workspace_compatibility.py \
      tests/integration/test_reconstruction_semantic_geometry_pipeline.py
    ;;
  backward)
    python -m pytest -q \
      tests/unit/test_phase9_1g_measurement_calibration.py \
      tests/unit/test_phase9_1g_inspection_calibration_provenance.py \
      tests/unit/test_phase8_8d5_offline_semantic_presentation.py
    ;;
  *)
    echo "Usage: $0 {quick|integration|backward}" >&2
    exit 2
    ;;
esac
