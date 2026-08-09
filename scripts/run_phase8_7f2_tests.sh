#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
case "${1:-quick}" in
  quick) python -m pytest -q tests/unit/test_reconstruction_da3.py tests/unit/test_reconstruction_da3_provider.py tests/unit/test_phase8_7f2_backward_compatibility.py ;;
  integration) python -m pytest -q tests/integration/test_reconstruction_da3_provider.py ;;
  stress) python -m pytest -q tests/stress/test_reconstruction_da3_provider_stress.py ;;
  *) echo "usage: $0 {quick|integration|stress}" >&2; exit 2 ;;
esac
