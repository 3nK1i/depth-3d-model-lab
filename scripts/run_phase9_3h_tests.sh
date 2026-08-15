#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mode="${1:-quick}"
case "$mode" in
  quick)
    pytest -q \
      tests/unit/test_phase9_3h_da3_canonical_scene.py \
      tests/unit/test_phase9_3h_da3_application_contract.py \
      tests/ui/test_phase9_3h_da3_canonical_integration.py
    ;;
  integration)
    pytest -q tests/integration/test_phase9_3h_da3_persistent_scene.py
    python scripts/run_phase9_3h_da3_canonical_smoke.py
    ;;
  backward)
    pytest -q \
      tests/unit/test_reconstruction_da3.py \
      tests/unit/test_reconstruction_da3_provider.py \
      tests/unit/test_reconstruction_pipeline_geometry_f3.py \
      tests/ui/test_phase9_3g_dense_provider_ui.py \
      tests/ui/test_phase9_3f_scene_layers.py
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
