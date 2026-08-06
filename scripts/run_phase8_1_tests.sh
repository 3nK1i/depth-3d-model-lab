#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
case "$mode" in
  quick)
    python -m pytest -q \
      tests/unit/test_reconstruction_camera_poses.py \
      tests/unit/test_reconstruction_contracts_session.py \
      tests/unit/test_reconstruction_sparse_events.py \
      tests/unit/test_reconstruction_profiles_backends.py \
      tests/unit/test_reconstruction_checkpoint_serialization.py \
      tests/unit/test_phase8_1_backward_compatibility.py \
      tests/unit/test_spatial_contracts.py \
      tests/unit/test_gpu_profiles.py \
      tests/unit/test_phase75_step10_backward_compatibility.py
    ;;
  integration)
    python -m pytest -q \
      tests/integration/test_reconstruction_contract_lifecycle.py \
      tests/stress/test_reconstruction_contracts_stress.py
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
