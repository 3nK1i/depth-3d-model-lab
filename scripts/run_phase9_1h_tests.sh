#!/usr/bin/env bash
set -euo pipefail

mode="${1:-quick}"

case "$mode" in
  quick)
    python -m pytest -q \
      tests/unit/test_phase9_1h_persistent_authority.py \
      tests/unit/test_reconstruction_persistent_scene_f5.py \
      tests/ui/test_phase9_1h_persisted_authority_ui.py \
      tests/ui/test_phase8_9c_persistence_reload_qualification.py
    ;;
  integration)
    python -m pytest -q \
      tests/integration/test_phase9_1h_persistence_reproducibility.py \
      tests/integration/test_phase8_9c_persistence_reproducibility.py
    ;;
  full)
    "$0" quick
    "$0" integration
    ;;
  *)
    echo "Usage: $0 {quick|integration|full}" >&2
    exit 2
    ;;
esac
