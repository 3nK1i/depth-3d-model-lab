#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run_quick() {
  python -m pytest -q \
    tests/unit/test_phase8_9f6_actual_application_acceptance.py \
    tests/integration/test_phase8_9f6_actual_application_contract.py
}

run_affected() {
  python -m pytest -q \
    tests/unit/test_phase8_9f6_actual_application_acceptance.py \
    tests/integration/test_phase8_9f6_actual_application_contract.py \
    tests/unit/test_phase8_9f5_persistent_multi_artifact_scene.py \
    tests/integration/test_phase8_9f5_persistent_real_scene_contract.py \
    tests/unit/test_phase8_9c_persistence_reproducibility.py \
    tests/integration/test_phase8_9c_persistence_reproducibility.py \
    tests/ui/test_phase8_9c_persistence_reload_qualification.py \
    tests/unit/test_phase8_8f2_reconstruction_gaussian_visualization.py \
    tests/unit/test_phase8_8f3_reconstruction_nerf_visualization.py \
    tests/unit/test_phase8_8f4_reconstruction_representation_switching.py \
    tests/unit/test_phase8_8f5_reconstruction_appearance_resources.py \
    tests/ui/test_phase8_8f2_gaussian_splatting_visualization.py \
    tests/ui/test_phase8_8f3_nerf_visualization.py \
    tests/ui/test_phase8_8f4_representation_switching.py \
    tests/ui/test_phase8_8f5_resource_fallback_ux.py \
    tests/ui/test_phase8_8g2_session_save_reload_interaction.py \
    tests/ui/test_phase8_8g6_full_reconstruction_ux_review.py \
    tests/unit/test_phase8_8h5_contract.py
}

case "$MODE" in
  quick) run_quick ;;
  affected) run_affected ;;
  *)
    echo "Usage: $0 {quick|affected}" >&2
    exit 2
    ;;
esac
