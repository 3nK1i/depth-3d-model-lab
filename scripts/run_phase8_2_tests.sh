#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
case "$mode" in
  quick)
    python -m pytest -q \
      tests/unit/test_reconstruction_ingestion_sources.py \
      tests/unit/test_reconstruction_ingestion_quality_sampling.py \
      tests/unit/test_reconstruction_ingestion_cache.py \
      tests/unit/test_reconstruction_ingestion_pipeline.py \
      tests/unit/test_phase8_2_backward_compatibility.py \
      tests/unit/test_reconstruction_camera_poses.py \
      tests/unit/test_reconstruction_contracts_session.py \
      tests/unit/test_reconstruction_profiles_backends.py \
      tests/unit/test_phase8_1_backward_compatibility.py
    ;;
  integration)
    python -m pytest -q \
      tests/integration/test_reconstruction_ingestion_lifecycle.py \
      tests/stress/test_reconstruction_ingestion_stress.py
    ;;
  full)
    for group in tests/unit tests/ui tests/integration tests/stress; do
      echo "==> pytest group: $group"
      python -m pytest -q "$group"
    done
    ;;
  *)
    echo "Usage: $0 {quick|integration|full}" >&2
    exit 2
    ;;
esac
