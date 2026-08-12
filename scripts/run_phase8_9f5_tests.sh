#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-quick}"
case "$MODE" in
  quick)
    pytest -q \
      tests/unit/test_phase8_9f5_persistent_multi_artifact_scene.py \
      tests/integration/test_phase8_9f5_persistent_real_scene_contract.py
    ;;
  affected)
    pytest -q \
      tests/unit/test_phase8_9f5_persistent_multi_artifact_scene.py \
      tests/integration/test_phase8_9f5_persistent_real_scene_contract.py \
      tests/unit/test_reconstruction_persistent_scene_f5.py \
      tests/integration/test_reconstruction_persistent_scene_f5.py \
      tests/stress/test_reconstruction_persistent_scene_f5_stress.py \
      tests/unit/test_phase8_9c_persistence_reproducibility.py \
      tests/integration/test_phase8_9c_persistence_reproducibility.py \
      tests/stress/test_phase8_9c_persistence_reproducibility_stress.py \
      tests/ui/test_phase8_9c_persistence_reload_qualification.py \
      tests/unit/test_phase8_8f2_reconstruction_gaussian_visualization.py \
      tests/unit/test_phase8_8f3_reconstruction_nerf_visualization.py \
      tests/unit/test_phase8_9f2_real_geometry_campaign.py \
      tests/integration/test_phase8_9f2_real_geometry_contract.py \
      tests/unit/test_phase8_9f3_real_gaussian_campaign.py \
      tests/integration/test_phase8_9f3_real_gaussian_contract.py \
      tests/unit/test_phase8_9f4_real_nerf_campaign.py \
      tests/integration/test_phase8_9f4_real_nerf_contract.py
    ;;
  *)
    echo "usage: $0 {quick|affected}" >&2
    exit 2
    ;;
esac
