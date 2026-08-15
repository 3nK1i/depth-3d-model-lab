#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mode="${1:-quick}"
case "$mode" in
  quick)
    pytest -q \
      tests/unit/test_phase9_3g_dense_provider_presentation.py \
      tests/ui/test_phase9_3g_dense_provider_ui.py
    ;;
  integration)
    pytest -q tests/integration/test_phase9_3g_dense_provider_ui_contract.py
    python scripts/run_phase9_3g_dense_provider_smoke.py
    ;;
  backward)
    pytest -q \
      tests/ui/test_phase9_3f_scene_layers.py \
      tests/ui/test_phase8_8f1_appearance_representation_controls.py \
      tests/ui/test_phase8_8f4_representation_switching.py \
      tests/ui/test_phase8_8f5_resource_fallback_ux.py
    ;;
  full)
    "$0" quick
    "$0" integration
    "$0" backward
    ;;
  *)
    echo "Usage: $0 {quick|integration|backward|full}" >&2
    exit 2
    ;;
esac
