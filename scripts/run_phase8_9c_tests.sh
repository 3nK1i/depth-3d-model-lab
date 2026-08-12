#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-quick}"

python -m compileall -q \
  core/reconstruction/persistent_scene.py \
  core/reconstruction/engine_checkpoint.py \
  app/reconstruction_persistence_controller.py \
  app/reconstruction_controller.py \
  scripts/run_phase8_9c_persistence_reproducibility_smoke.py \
  tests/unit/test_phase8_9c_persistence_reproducibility.py \
  tests/integration/test_phase8_9c_persistence_reproducibility.py \
  tests/stress/test_phase8_9c_persistence_reproducibility_stress.py \
  tests/ui/test_phase8_9c_persistence_reload_qualification.py

case "$MODE" in
  quick)
    python -m pytest -q \
      tests/unit/test_phase8_9c_persistence_reproducibility.py \
      tests/integration/test_phase8_9c_persistence_reproducibility.py \
      tests/unit/test_reconstruction_persistent_scene_f5.py \
      tests/integration/test_reconstruction_persistent_scene_f5.py \
      tests/unit/test_reconstruction_engine_checkpoint_g2.py \
      tests/integration/test_reconstruction_engine_checkpoint_g2.py
    ;;
  affected)
    python -m pytest -q \
      tests/unit/test_phase8_9c_persistence_reproducibility.py \
      tests/integration/test_phase8_9c_persistence_reproducibility.py \
      tests/stress/test_phase8_9c_persistence_reproducibility_stress.py \
      tests/ui/test_phase8_9c_persistence_reload_qualification.py \
      tests/unit/test_reconstruction_persistent_scene_f5.py \
      tests/integration/test_reconstruction_persistent_scene_f5.py \
      tests/stress/test_reconstruction_persistent_scene_f5_stress.py \
      tests/unit/test_reconstruction_engine_checkpoint_g2.py \
      tests/integration/test_reconstruction_engine_checkpoint_g2.py \
      tests/stress/test_reconstruction_engine_checkpoint_g2_stress.py \
      tests/unit/test_phase8_8a4_reconstruction_persistence_controller.py \
      tests/integration/test_phase8_8a4_scene_checkpoint_bridge.py \
      tests/unit/test_phase8_8g2_reconstruction_session_interaction.py \
      tests/ui/test_phase8_8g2_session_save_reload_interaction.py \
      tests/unit/test_phase8_8g3_reconstruction_checkpoint_resume.py \
      tests/ui/test_phase8_8g3_checkpoint_resume_ux.py \
      tests/unit/test_phase8_9a1_reconstruction_export.py \
      tests/integration/test_phase8_9a1_reconstruction_export.py \
      tests/unit/test_phase8_9a2_reconstruction_export_ui_contract.py \
      tests/unit/test_phase8_9b_qc_export_contract.py
    ;;
  *)
    echo "Usage: $0 {quick|affected}" >&2
    exit 2
    ;;
esac
